#!/usr/bin/env python3
"""
scripts/research/verify_recoverability.py — Blind answer-recoverability audit.

Implements the stratified sampling, blind prompting (gold answers hidden),
tolerant JSON choice parsing, and summary statistics (Brennan-Prediger kappa,
bootstrap 95% CIs, and refusal rates stratified by gold lexical coverage)
reported in Section 4.6 and Appendix B of the HOEIT-LegalQA paper.

Endpoint credentials are read strictly from standard environment variables
(``OPENAI_API_BASE`` / ``OPENAI_API_KEY`` or CLI flags).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mcq_utils import resolve_ground_truth_index, strip_choice_label

VALID_CHOICES = ("A", "B", "C", "D")

SYSTEM_PROMPT = (
    "Bạn là một chuyên gia pháp luật Việt Nam, thẩm định độc lập một bộ câu hỏi trắc nghiệm luật. "
    "Bạn chỉ được dựa vào ngữ cảnh được cung cấp, không tra cứu ngoài. "
    "Trả lời bằng tiếng Việt. Chỉ xuất một khối JSON, không kèm chữ nào khác ngoài JSON."
)

INSTRUCTION_PROMPT = """Với MỖI câu dưới đây: chỉ dựa vào phần NGỮ CẢNH và CÂU HỎI, hãy tự chọn đáp án đúng nhất trong 4 phương án A/B/C/D.
Nếu ngữ cảnh KHÔNG đủ căn cứ để chọn, đặt choice = "KHONG_DU_CAN_CU" và nói rõ thiếu gì. Không đoán bừa.
Trong 3 phương án còn lại, nếu có phương án thứ hai cũng đúng về mặt pháp lý thì đặt multiple_correct_risk = true.

Trả về JSON đúng định dạng, đủ số câu được giao:
{"answers":[{"stt":1,"choice":"B","confidence":0.9,"multiple_correct_risk":false,"ly_do_ngan":"1-2 câu tiếng Việt"}]}"""

REFUSAL_MARKERS = (
    "KHONG_DU_CAN_CU",
    "KHÔNG ĐỦ CĂN CỨ",
    "KHONG DU CAN CU",
    "KHÔNG ĐỦ CĂN CỨ ĐỂ",
    "KHÔNG HỢP LỆ",
    "KHONG HOP LE",
    "KHÔNG XÁC ĐỊNH ĐƯỢC",
    "KHONG XAC DINH DUOC",
    "KHÔNG THỂ XÁC ĐỊNH",
    "KHÔNG CHỌN ĐƯỢC",
    "KHÔNG CÓ ĐÁP ÁN",
    "KHÔNG CÓ CĂN CỨ",
    "THIẾU CĂN CỨ",
    "KHÔNG ĐỦ THÔNG TIN",
    "KHÔNG ĐỦ DỮ KIỆN",
)

# Priority-ordered keys that name the chosen option letter/text.
OPTION_KEYS = (
    "choice",
    "correct_option",
    "correct_choice",
    "selected_option",
    "đáp án đúng",
    "đáp án chính xác",
    "đáp án đúng nhất",
    "đáp án được chọn",
    "đáp án chọn",
    "đáp án của đề",
    "đáp án cuối cùng",
    "đáp án dự kiến",
    "đáp án đã chọn",
    "đáp án",
    "phương án đúng",
    "phương án chọn",
    "dap_an_dung",
    "dap_an_chinh_xac",
    "dap_an_chon",
    "dap_an_cua_de",
    "dap_an_du_kien",
    "dap_an",
    "dapan",
    "câu trả lời",
    "cau_tra_loi",
    "câu tl",
    "kết luận",
    "ket_luan",
    "đáp án đúng là",
    "chọn",
    "answer",
)

# Keys carrying context/question/rationale metadata rather than the final label.
CONTENT_KEYS = frozenset({
    "câu hỏi",
    "cau hoi",
    "câu hỏi thi",
    "noi dung cau hoi",
    "nội dung câu hỏi",
    "phương án",
    "phuong an",
    "phuong_an",
    "ngữ cảnh",
    "ngu canh",
    "noi dung",
    "nội dung",
    "lý do",
    "ly do",
    "ly_do",
    "ly do ngan",
    "ly_do_ngan",
    "giải thích",
    "giai thich",
    "giải_thích",
    "explanation",
    "reason",
    "text",
    "ghi chú",
    "ghi_chu",
    "đánh giá",
    "danh_gia",
    "phân tích",
    "phan_tich",
    "độ tin cậy",
    "mức độ tin cậy",
    "do_tin_cay",
    "muc_do_tin_cay",
    "confidence",
    "điểm",
    "diem",
    "số",
    "nguồn",
    "nguon",
    "stt",
    "index",
})

_NORMALIZED_CONTENT_KEYS: frozenset[str] | None = None
_WRITE_LOCK = threading.Lock()


def _get_context_text(record: dict[str, Any]) -> str:
    """Read context text across both flat HF JSONL and nested pipeline schemas."""
    if record.get("context_text"):
        return str(record["context_text"])
    payload = record.get("context_payload")
    if isinstance(payload, dict) and payload.get("text"):
        return str(payload["text"])
    return ""


def _norm_key(key: Any) -> str:
    text = unicodedata.normalize("NFC", str(key)).strip().lower()
    return re.sub(r"[\s_\-\.]+", " ", text)


def _get_normalized_content_keys() -> frozenset[str]:
    global _NORMALIZED_CONTENT_KEYS
    if _NORMALIZED_CONTENT_KEYS is None:
        _NORMALIZED_CONTENT_KEYS = frozenset(_norm_key(k) for k in CONTENT_KEYS)
    return _NORMALIZED_CONTENT_KEYS


def _strip_diacritics(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def _slug(text: str) -> str:
    flat = _strip_diacritics(str(text)).lower()
    flat = re.sub(r"[^a-z0-9 ]+", " ", flat)
    return re.sub(r"\s+", " ", flat).strip()


def _as_text(value: Any) -> str:
    if value is None or isinstance(value, (list, dict, bool)):
        return ""
    return str(value).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_stratified_sample(
    records: Sequence[dict[str, Any]],
    *,
    total: int = 250,
    floor: int = 5,
    top_domains: int = 10,
    seed: int = 20261120,
    deduplicate_pass2: bool = True,
) -> list[dict[str, Any]]:
    """Draw a domain x Bloom stratified sample matching Section 4.6."""
    domain_counts = collections.Counter(r.get("domain_tag") for r in records if r.get("domain_tag"))
    ordered_domains = [d for d, _ in domain_counts.most_common()]
    buckets = ordered_domains[:top_domains]
    bucket_of = {d: (d if d in buckets else buckets[-1]) for d in ordered_domains}

    cells: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for rec in records:
        bloom = rec.get("bloom_level")
        domain = bucket_of.get(rec.get("domain_tag", ""))
        if bloom and domain:
            cells[(domain, str(bloom))].append(rec)

    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for key in sorted(cells):
        pool = cells[key][:]
        rng.shuffle(pool)
        for rec in pool[:floor]:
            sample.append(rec)
            seen_ids.add(id(rec))

    remaining = total - len(sample)
    if remaining > 0:
        headroom: list[dict[str, Any]] = []
        for key in sorted(cells):
            if deduplicate_pass2:
                pool = [rec for rec in cells[key] if id(rec) not in seen_ids]
                rng.shuffle(pool)
                headroom.extend(pool)
            else:
                taken = sum(1 for s in sample if (s.get("domain_tag"), s.get("bloom_level")) == key)
                pool = cells[key][:]
                rng.shuffle(pool)
                for rec in pool:
                    if taken < len(cells[key]):
                        headroom.append(rec)
                        taken += 1
        rng.shuffle(headroom)
        sample.extend(headroom[:remaining])

    rng.shuffle(sample)
    return sample[:total]


def build_batch_prompt(batch: Sequence[dict[str, Any]]) -> str:
    """Build the blind verification user prompt for 1 or more items (no gold answer)."""
    parts: list[str] = []
    for stt, rec in enumerate(batch, start=1):
        candidates = rec.get("candidate_answers") or []
        lines = [
            f"STT: {stt}",
            "NGỮ CẢNH:",
            _get_context_text(rec),
            "CÂU HỎI:",
            str(rec.get("question_content", "")),
            "PHƯƠNG ÁN:",
        ]
        for label, cand in zip(VALID_CHOICES, candidates):
            clean = strip_choice_label(str(cand)) or str(cand).strip()
            lines.append(f"{label}. {clean}")
        parts.append("\n".join(lines))
    return INSTRUCTION_PROMPT + "\n\n---\n\n" + "\n\n---\n\n".join(parts)


def extract_json_payload(text: str | None) -> Any:
    """Extract the first valid JSON object/array from a model response."""
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    b_start, b_end = cleaned.find("["), cleaned.rfind("]")
    if b_start != -1 and (start == -1 or b_start < start):
        start, end = b_start, b_end
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def _letter_from_value(value: Any) -> str | None:
    text = _as_text(value)
    if not text:
        return None
    flat = _strip_diacritics(text).upper()
    for marker in REFUSAL_MARKERS:
        if _strip_diacritics(marker).upper() in flat:
            return "REFUSED"
    match = re.match(r"^[\s\(\[\{\"']*([A-D])(?:\s*[\.\)\:\,\-\]\}\"']|\s|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if len(text) <= 12:
        short = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
        if short:
            return short.group(1).upper()
    return None


def _match_candidate(text: str, candidates: Sequence[str]) -> str | None:
    query = _slug(text)
    if not query or len(query) < 8:
        return None
    best_idx: int | None = None
    best_score = 0.0
    query_tokens = set(query.split())
    for idx, cand in enumerate(candidates[:4]):
        cand_slug = _slug(strip_choice_label(str(cand)) or str(cand))
        if not cand_slug:
            continue
        if query == cand_slug or (cand_slug.startswith(query) and len(query) >= 15):
            return VALID_CHOICES[idx]
        cand_tokens = set(cand_slug.split())
        if not query_tokens or not cand_tokens:
            continue
        overlap = len(query_tokens & cand_tokens) / len(query_tokens)
        if overlap > best_score:
            best_score, best_idx = overlap, idx
    if best_idx is not None and best_score >= 0.60:
        return VALID_CHOICES[best_idx]
    return None


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("answers", "ket_qua", "results", "data", "kết quả"):
            val = payload.get(key)
            if isinstance(val, list):
                return [item for item in val if isinstance(item, dict)]
        return [payload]
    return []


def _extract_risk_flag(item: dict[str, Any]) -> bool | None:
    for key in ("multiple_correct_risk", "multiple_correct"):
        if key in item and isinstance(item[key], bool):
            return item[key]
    return None


def _parse_item_choice(item: dict[str, Any], candidates: Sequence[str]) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    for key, val in item.items():
        lookup.setdefault(_norm_key(key), val)

    for opt_key in OPTION_KEYS:
        nk = _norm_key(opt_key)
        if nk in lookup:
            raw = _as_text(lookup[nk])
            got = _letter_from_value(lookup[nk])
            if got is None and raw:
                got = _match_candidate(raw, candidates)
            if got:
                return {"choice": got, "field": opt_key, "raw_choice": raw, "multiple_correct_risk": _extract_risk_flag(item)}

    content_keys = _get_normalized_content_keys()
    for key, val in item.items():
        if _norm_key(key) in content_keys:
            continue
        got = _letter_from_value(val)
        if got:
            return {"choice": got, "field": key, "raw_choice": _as_text(val), "multiple_correct_risk": _extract_risk_flag(item)}

    for opt_key in OPTION_KEYS:
        nk = _norm_key(opt_key)
        if nk in lookup:
            raw = _as_text(lookup[nk])
            got = _match_candidate(raw, candidates)
            if got:
                return {"choice": got, "field": opt_key, "raw_choice": raw, "multiple_correct_risk": _extract_risk_flag(item)}

    for key, val in item.items():
        if _norm_key(key) in content_keys:
            continue
        raw = _as_text(val)
        got = _match_candidate(raw, candidates)
        if got:
            return {"choice": got, "field": key, "raw_choice": raw, "multiple_correct_risk": _extract_risk_flag(item)}

    return {"choice": None, "field": None, "raw_choice": "", "multiple_correct_risk": _extract_risk_flag(item)}


def parse_choice(payload: Any, candidates: Sequence[str]) -> dict[str, Any]:
    """Parse a single-item verification response into A/B/C/D, REFUSED, or None."""
    items = _extract_items(payload)
    if not items:
        return {"choice": None, "field": None, "raw_choice": "", "multiple_correct_risk": None}
    return _parse_item_choice(items[0], candidates)


def parse_batch_choices(payload: Any, batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve item predictions for a batch of 1 or more questions by stt or position."""
    items = _extract_items(payload)
    by_stt: dict[int, dict[str, Any]] = {}
    for pos, item in enumerate(items, start=1):
        stt_raw = item.get("stt", item.get("STT", pos))
        try:
            stt = int(stt_raw)
        except (TypeError, ValueError):
            stt = pos
        by_stt.setdefault(stt, item)

    resolved: list[dict[str, Any]] = []
    for pos, rec in enumerate(batch, start=1):
        item = by_stt.get(pos) or (items[pos - 1] if pos - 1 < len(items) else None)
        if item is None:
            resolved.append({"choice": None, "field": None, "raw_choice": "", "multiple_correct_risk": None})
        else:
            resolved.append(_parse_item_choice(item, rec.get("candidate_answers") or []))
    return resolved


def resolve_gold_letter(record: dict[str, Any]) -> str | None:
    """Resolve the gold option letter (A-D) for post-hoc evaluation only."""
    gl = record.get("gold_letter")
    if gl and str(gl).strip().upper()[:1] in VALID_CHOICES:
        return str(gl).strip().upper()[:1]
    idx = record.get("gold_index")
    if isinstance(idx, int) and 0 <= idx < 4:
        return VALID_CHOICES[idx]
    res = resolve_ground_truth_index(record.get("candidate_answers") or [], str(record.get("ground_truth", "")))
    if res["matched_index"] is not None and 0 <= res["matched_index"] < 4:
        return VALID_CHOICES[res["matched_index"]]
    return None


def lexical_coverage(context: str, gold_answer: str) -> float:
    """Compute share of gold-answer tokens (>1 char) present in the context string."""
    if not gold_answer:
        return 0.0
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in gold_answer)
    tokens = [tok for tok in cleaned.split() if len(tok) > 1]
    if not tokens:
        return 0.0
    ctx_lower = context.lower()
    return sum(1 for tok in tokens if tok.lower() in ctx_lower) / len(tokens)


def brennan_prediger_kappa(matched: int, total: int, n_categories: int = 4) -> float | None:
    """Chance-corrected agreement assuming uniform random baseline p_e = 1/k."""
    if total <= 0 or n_categories <= 1:
        return None
    p_o = matched / total
    p_e = 1.0 / n_categories
    return round((p_o - p_e) / (1.0 - p_e), 2)


def bootstrap_proportion_ci(
    successes: int,
    total: int,
    *,
    n_boot: int = 10000,
    seed: int = 20261120,
) -> tuple[float | None, float | None]:
    """Percentile 95% bootstrap confidence interval in percent."""
    if total <= 0:
        return (None, None)
    flags = [1] * successes + [0] * (total - successes)
    rng = random.Random(seed)
    estimates = sorted(sum(rng.choices(flags, k=total)) / total for _ in range(n_boot))
    low = round(100.0 * estimates[int(0.025 * n_boot)], 1)
    high = round(100.0 * estimates[min(n_boot - 1, int(0.975 * n_boot))], 1)
    return (low, high)


def summarize_results(
    records: Sequence[dict[str, Any]],
    rows_by_index: dict[int, dict[str, Any]],
    *,
    model: str,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Aggregate accuracy, refusal rate, Brennan-Prediger kappa, and coverage slices."""
    matched = refused = answered = risk = 0
    choice_dist: collections.Counter[str] = collections.Counter()
    refuse_by_cov = {k: [0, 0] for k in ("ge080", "050_079", "lt050")}
    context_lengths: dict[str, list[int]] = {"ge080": [], "050_079": [], "lt050": []}

    for idx, rec in enumerate(records):
        ctx = _get_context_text(rec)
        cov = lexical_coverage(ctx, str(rec.get("ground_truth", "")))
        bucket = "ge080" if cov >= 0.80 else ("050_079" if cov >= 0.50 else "lt050")
        context_lengths[bucket].append(len(ctx))

        row = rows_by_index.get(idx)
        if not row:
            continue
        choice = str(row.get("choice") or "").strip().upper()
        if choice == "REFUSED":
            refused += 1
            refuse_by_cov[bucket][1] += 1
            continue
        if choice not in VALID_CHOICES:
            continue

        answered += 1
        refuse_by_cov[bucket][0] += 1
        choice_dist[choice] += 1
        risk_flag = row.get("multiple_correct_risk")
        if risk_flag is None and row.get("raw"):
            items = _extract_items(extract_json_payload(row.get("raw")))
            risk_flag = _extract_risk_flag(items[0]) if items else None
        if risk_flag is True:
            risk += 1
        if choice == resolve_gold_letter(rec):
            matched += 1

    n_usable = answered + refused
    return {
        "model": model,
        "batch_size": batch_size,
        "seed": seed,
        "n_sample": len(records),
        "n_usable": n_usable,
        "matched_gold": matched,
        "matched_pct": round(100.0 * matched / n_usable, 1) if n_usable else None,
        "matched_ci95": list(bootstrap_proportion_ci(matched, n_usable, seed=seed)),
        "refused": refused,
        "refused_pct": round(100.0 * refused / n_usable, 1) if n_usable else None,
        "refused_ci95": list(bootstrap_proportion_ci(refused, n_usable, seed=seed)),
        "answered": answered,
        "matched_among_answered_pct": round(100.0 * matched / answered, 1) if answered else None,
        "matched_among_answered_ci95": list(bootstrap_proportion_ci(matched, answered, seed=seed)),
        "brennan_prediger_kappa_answered": brennan_prediger_kappa(matched, answered),
        "brennan_prediger_kappa_all": brennan_prediger_kappa(matched, n_usable),
        "multiple_correct_risk": risk,
        "multiple_correct_risk_pct_all": round(100.0 * risk / n_usable, 1) if n_usable else None,
        "multiple_correct_risk_pct_answered": round(100.0 * risk / answered, 1) if answered else None,
        "choice_distribution_answered": dict(choice_dist),
        "refusal_by_coverage": {
            bucket: {
                "answered": vals[0],
                "refused": vals[1],
                "n": vals[0] + vals[1],
                "refused_pct": round(100.0 * vals[1] / (vals[0] + vals[1]), 1)
                if (vals[0] + vals[1])
                else None,
                "median_context_chars": round(statistics.median(context_lengths[bucket]))
                if context_lengths[bucket]
                else None,
            }
            for bucket, vals in refuse_by_cov.items()
        },
    }


def _call_chat_completions(api_base: str, api_key: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    req = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _verify_batch(
    indices: Sequence[int],
    batch: Sequence[dict[str, Any]],
    *,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    attempts: int,
    results_out: Path | None,
) -> list[dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_batch_prompt(batch)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_error = "unknown"
    for attempt in range(attempts):
        try:
            response = _call_chat_completions(api_base, api_key, payload)
            content = response["choices"][0]["message"]["content"]
            parsed_items = parse_batch_choices(extract_json_payload(content), batch)
            if all(p["choice"] is not None for p in parsed_items):
                rows = [
                    {
                        "index": idx,
                        "qa_id": rec.get("qa_id"),
                        "choice": parsed["choice"],
                        "field": parsed["field"],
                        "raw_choice": parsed["raw_choice"][:300],
                        "multiple_correct_risk": parsed["multiple_correct_risk"],
                        "attempts": attempt + 1,
                        "usage": response.get("usage"),
                        "raw": content,
                    }
                    for idx, rec, parsed in zip(indices, batch, parsed_items)
                ]
                if results_out:
                    for row in rows:
                        _append_jsonl(results_out, row)
                return rows
            last_error = "incomplete_or_unparsed_response"
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001
            last_error = type(exc).__name__
        time.sleep(min(60, 3 * (attempt + 1) ** 2))

    rows = [
        {"index": idx, "qa_id": rec.get("qa_id"), "choice": None, "error": last_error}
        for idx, rec in zip(indices, batch)
    ]
    if results_out:
        for row in rows:
            _append_jsonl(results_out, row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Blind answer-recoverability verification on stratified test samples."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input test JSONL or pre-sampled JSONL")
    parser.add_argument("--sample-out", type=Path, default=None, help="Optional path to save sampled records")
    parser.add_argument("--results-out", type=Path, default=None, help="JSONL file for per-item predictions")
    parser.add_argument("--summary-out", type=Path, default=None, help="JSON file for aggregated summary")
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--cell-floor", type=int, default=5)
    parser.add_argument("--top-domains", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument(
        "--legacy-pass2-resample",
        action="store_true",
        help="Reproduce the archived pass-2 reshuffle without subtracting pass-1 items",
    )
    parser.add_argument("--already-sampled", action="store_true", help="Treat --input as the final sample")
    parser.add_argument("--sample-only", action="store_true", help="Write stratified sample and exit")
    parser.add_argument("--analyze-only", action="store_true", help="Compute summary from --results-out")
    parser.add_argument("--model", type=str, default="", help="OpenAI-compatible model ID")
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("OPENAI_API_BASE") or os.getenv("BENCHMARK_API_BASE", ""),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY") or os.getenv("BENCHMARK_API_KEY", ""),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=20, help="20 for batched mode; 1 for per-item calls")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args(argv)

    records = load_jsonl(args.input)
    if not args.already_sampled:
        records = build_stratified_sample(
            records,
            total=args.sample_size,
            floor=args.cell_floor,
            top_domains=args.top_domains,
            seed=args.seed,
            deduplicate_pass2=not args.legacy_pass2_resample,
        )
    if args.sample_out:
        args.sample_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.sample_out, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if args.sample_only:
        print(f"[OK] Wrote {len(records)} stratified items")
        return 0

    rows_by_index: dict[int, dict[str, Any]] = {}
    if args.results_out and args.results_out.exists():
        for row in load_jsonl(args.results_out):
            if row.get("choice") is not None and isinstance(row.get("index"), int):
                rows_by_index[row["index"]] = row

    if not args.analyze_only:
        if not args.model or not args.api_base or not args.api_key:
            raise SystemExit("Require --model, --api-base (or OPENAI_API_BASE), and --api-key (or OPENAI_API_KEY).")
        todo = [(idx, rec) for idx, rec in enumerate(records) if idx not in rows_by_index]
        batches = [todo[pos : pos + args.batch_size] for pos in range(0, len(todo), args.batch_size)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    _verify_batch,
                    [idx for idx, _ in chunk],
                    [rec for _, rec in chunk],
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    attempts=args.attempts,
                    results_out=args.results_out,
                )
                for chunk in batches
            ]
            for fut in as_completed(futures):
                for row in fut.result():
                    rows_by_index[row["index"]] = row

    summary = summarize_results(
        records,
        rows_by_index,
        model=args.model or "unspecified",
        batch_size=args.batch_size,
        seed=args.seed,
    )
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
