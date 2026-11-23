"""Tests for stratified sampling, blind prompt isolation, and response parsing."""

from __future__ import annotations

import json

from verify_recoverability import (
    INSTRUCTION_PROMPT,
    SYSTEM_PROMPT,
    brennan_prediger_kappa,
    build_batch_prompt,
    build_stratified_sample,
    extract_json_payload,
    lexical_coverage,
    parse_batch_choices,
    parse_choice,
    summarize_results,
)

SAMPLE_CANDIDATES = [
    "Quốc hội quyết định thành lập.",
    "Chính phủ ban hành nghị định hướng dẫn.",
    "Ủy ban nhân dân cấp tỉnh ban hành quyết định.",
    "Tòa án nhân dân tối cao giải thích.",
]


class TestBlindPromptIsolation:
    def test_prompt_hides_gold_answer_and_rationale(self):
        secret_gt = "SECRET_GOLD_TOKEN_DO_NOT_LEAK"
        secret_rationale = "SECRET_RATIONALE_DO_NOT_LEAK"
        record = {
            "qa_id": "doc1_remember_0",
            "context_text": "Điều 5 quy định cơ quan có thẩm quyền là Chính phủ.",
            "question_content": "Cơ quan nào có thẩm quyền ban hành nghị định?",
            "candidate_answers": SAMPLE_CANDIDATES,
            "ground_truth": secret_gt,
            "gold_index": 1,
            "gold_letter": "B",
            "legal_rationale": secret_rationale,
        }
        prompt = build_batch_prompt([record])
        assert INSTRUCTION_PROMPT in prompt
        assert "STT: 1" in prompt
        assert "B. Chính phủ ban hành nghị định hướng dẫn." in prompt
        assert secret_gt not in prompt
        assert secret_rationale not in prompt
        assert "KHONG_DU_CAN_CU" in SYSTEM_PROMPT or "KHONG_DU_CAN_CU" in prompt


class TestChoiceParsing:
    def test_parses_standard_answers_schema(self):
        raw = '{"answers":[{"stt":1,"choice":"B","confidence":0.95,"multiple_correct_risk":false,"ly_do_ngan":"Rõ căn cứ."}]}'
        parsed = parse_choice(extract_json_payload(raw), SAMPLE_CANDIDATES)
        assert parsed["choice"] == "B"
        assert parsed["field"] == "choice"
        assert parsed["multiple_correct_risk"] is False

    def test_parses_refusal_only_from_option_field(self):
        raw = json.dumps(
            {
                "choice": "KHONG_DU_CAN_CU",
                "ly_do_ngan": "Ngữ cảnh thiếu thông tin cần thiết.",
            },
            ensure_ascii=False,
        )
        parsed = parse_choice(extract_json_payload(raw), SAMPLE_CANDIDATES)
        assert parsed["choice"] == "REFUSED"

        non_refusal = json.dumps(
            {
                "Đáp án đúng": "B",
                "lý do": "Dù có điểm hơi thiếu căn cứ nhưng B vẫn đúng nhất.",
            },
            ensure_ascii=False,
        )
        parsed_non_refusal = parse_choice(extract_json_payload(non_refusal), SAMPLE_CANDIDATES)
        assert parsed_non_refusal["choice"] == "B"

    def test_parses_vietnamese_key_or_text_match(self):
        raw = '```json\n{"Đáp án chọn":"Ủy ban nhân dân cấp tỉnh ban hành quyết định."}\n```'
        parsed = parse_choice(extract_json_payload(raw), SAMPLE_CANDIDATES)
        assert parsed["choice"] == "C"

    def test_parses_batched_payload_by_stt(self):
        batch = [
            {"candidate_answers": SAMPLE_CANDIDATES},
            {"candidate_answers": SAMPLE_CANDIDATES},
        ]
        payload = {
            "answers": [
                {"stt": 2, "choice": "D", "multiple_correct_risk": True},
                {"stt": 1, "choice": "A", "multiple_correct_risk": False},
            ]
        }
        resolved = parse_batch_choices(payload, batch)
        assert [r["choice"] for r in resolved] == ["A", "D"]
        assert [r["multiple_correct_risk"] for r in resolved] == [False, True]


class TestStratificationAndMetrics:
    def test_stratified_sample_is_deterministic_and_honors_size(self):
        records = []
        domains = [f"Domain {i}" for i in range(12)]
        blooms = ("Remember", "Understand", "Apply")
        for d in domains:
            for b in blooms:
                for k in range(12):
                    records.append({"qa_id": f"{d}_{b}_{k}", "domain_tag": d, "bloom_level": b})

        s1 = build_stratified_sample(records, total=250, floor=5, top_domains=10, seed=20261120)
        s2 = build_stratified_sample(records, total=250, floor=5, top_domains=10, seed=20261120)
        assert len(s1) == 250
        assert len({r["qa_id"] for r in s1}) == 250
        assert [r["qa_id"] for r in s1] == [r["qa_id"] for r in s2]

    def test_brennan_prediger_and_summary_metrics(self):
        assert brennan_prediger_kappa(123, 195) == 0.51
        assert brennan_prediger_kappa(154, 230) == 0.56
        assert lexical_coverage("Chính phủ ban hành nghị định hướng dẫn thi hành.", "Chính phủ ban hành nghị định") == 1.0

        records = [
            {"context_text": "Chính phủ ban hành nghị định", "ground_truth": "Chính phủ ban hành nghị định", "gold_index": 1},
            {"context_text": "Đoạn rất ngắn", "ground_truth": "Một đáp án dài không có trong đoạn trích", "gold_index": 0},
        ]
        rows = {
            0: {"choice": "B", "raw": '{"answers":[{"stt":1,"choice":"B","multiple_correct_risk":true}]}'},
            1: {"choice": "REFUSED", "multiple_correct_risk": False},
        }
        summary = summarize_results(records, rows, model="test-model", batch_size=1, seed=42)
        assert summary["n_usable"] == 2
        assert summary["matched_gold"] == 1
        assert summary["refused"] == 1
        assert summary["multiple_correct_risk"] == 1
        assert summary["refusal_by_coverage"]["ge080"]["refused"] == 0
        assert summary["refusal_by_coverage"]["lt050"]["refused"] == 1
