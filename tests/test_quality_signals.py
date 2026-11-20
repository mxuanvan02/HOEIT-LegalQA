"""Tests for deterministic per-record quality signals and triage.

`compute_quality_signals` is the filter gate for the released dataset. It
must stay deterministic (no LLM, no clock, no randomness) so that a rebuild
produces the same triage decisions, and it must never silently pass a record
whose ground truth cannot be resolved.
"""

from __future__ import annotations

from quality_signals import (
    QUALITY_KEY,
    QUALITY_SCHEMA_VERSION,
    QualityThresholds,
    compute_quality_signals,
    stamp_quality,
)

GOOD_RECORD = {
    "question_content": "Theo Bộ luật Tố tụng Hình sự, ai có thẩm quyền phê chuẩn lệnh bắt?",
    "candidate_answers": [
        "A. Viện kiểm sát.",
        "B. Tòa án.",
        "C. Cơ quan điều tra.",
        "D. Ủy ban nhân dân.",
    ],
    "ground_truth": "A. Viện kiểm sát.",
    "legal_rationale": (
        "Đại tiền đề: theo quy định của Bộ luật Tố tụng Hình sự. "
        "Tiểu tiền đề: trong trường hợp bắt bị can để tạm giam. "
        "Kết luận: Viện kiểm sát phê chuẩn lệnh bắt."
    ),
    "context_text": (
        "Điều 113. Việc bắt bị can để tạm giam phải có phê chuẩn của "
        "Viện kiểm sát cùng cấp trước khi thi hành, còn các trường hợp "
        "khác do cơ quan điều tra thực hiện theo quy định của pháp luật."
    ),
    "is_multimodal": False,
}


def _codes(result):
    return {flag["code"] for flag in result["flags"]}


class TestGoodRecord:
    def test_resolves_ground_truth(self):
        result = compute_quality_signals(GOOD_RECORD)
        assert result["metrics"]["gt_matched_index"] == 0

    def test_no_error_severity_flags(self):
        result = compute_quality_signals(GOOD_RECORD)
        errors = [f for f in result["flags"] if f["severity"] == "error"]
        assert errors == [], [f["code"] for f in errors]

    def test_triage_is_not_regenerate(self):
        assert compute_quality_signals(GOOD_RECORD)["triage"] in ("keep", "review")

    def test_schema_version_reported(self):
        assert compute_quality_signals(GOOD_RECORD)["schema"] == QUALITY_SCHEMA_VERSION

    def test_is_deterministic(self):
        first = compute_quality_signals(GOOD_RECORD)
        for _ in range(3):
            assert compute_quality_signals(GOOD_RECORD) == first


class TestRegenerateTriggers:
    def test_unresolvable_ground_truth_forces_regenerate(self):
        record = dict(GOOD_RECORD, ground_truth="Một đáp án hoàn toàn không có trong danh sách")
        result = compute_quality_signals(record)
        assert "gt_unresolved" in _codes(result)
        assert result["triage"] == "regenerate"

    def test_duplicate_candidates_force_regenerate(self):
        record = dict(
            GOOD_RECORD,
            candidate_answers=["A. Viện kiểm sát.", "A. Viện kiểm sát.", "C. Cơ quan.", "D. Ủy ban."],
        )
        result = compute_quality_signals(record)
        assert "duplicate_candidates" in _codes(result)
        assert result["triage"] == "regenerate"

    def test_prompt_leak_is_an_error(self):
        record = dict(
            GOOD_RECORD,
            legal_rationale="<qa_pair> 根据提供的背景信息，答案是A。 Kết luận: A.",
        )
        result = compute_quality_signals(record)
        assert "language_artifact" in _codes(result)
        assert result["triage"] == "regenerate"

    def test_judge_ungrounded_forces_regenerate(self):
        record = dict(GOOD_RECORD, eval_scores={"groundedness": 0.0})
        assert compute_quality_signals(record)["triage"] == "regenerate"


class TestReviewTriggers:
    def test_short_question_is_flagged(self):
        record = dict(GOOD_RECORD, question_content="Ai?")
        assert "question_too_short" in _codes(compute_quality_signals(record))

    def test_short_context_is_flagged(self):
        record = dict(GOOD_RECORD, context_text="ngắn")
        assert "context_too_short" in _codes(compute_quality_signals(record))

    def test_missing_syllogism_is_flagged(self):
        record = dict(GOOD_RECORD, legal_rationale="Vì luật quy định vậy nên đáp án là A.")
        assert "missing_syllogism" in _codes(compute_quality_signals(record))

    def test_multimodal_without_visual_is_flagged(self):
        record = dict(GOOD_RECORD, is_multimodal=True, context_visuals=[])
        assert "multimodal_no_visual" in _codes(compute_quality_signals(record))

    def test_wrong_candidate_count_is_flagged(self):
        record = dict(GOOD_RECORD, candidate_answers=["A. Viện kiểm sát.", "B. Tòa án."])
        assert "candidate_count" in _codes(compute_quality_signals(record))


class TestStampQuality:
    def test_stamp_attaches_under_reserved_key(self):
        record = dict(GOOD_RECORD)
        returned = stamp_quality(record)
        assert QUALITY_KEY in returned
        assert returned[QUALITY_KEY]["schema"] == QUALITY_SCHEMA_VERSION

    def test_stamp_mutates_in_place_and_returns_same_object(self):
        record = dict(GOOD_RECORD)
        assert stamp_quality(record) is record

    def test_stamp_overwrites_previous_stamp(self):
        record = dict(GOOD_RECORD)
        stamp_quality(record)
        first = record[QUALITY_KEY]
        stamp_quality(record)
        assert record[QUALITY_KEY] == first


class TestThresholds:
    def test_custom_thresholds_change_the_verdict(self):
        strict = QualityThresholds(min_question_chars=500)
        assert "question_too_short" in _codes(compute_quality_signals(GOOD_RECORD, strict))

    def test_default_thresholds_keep_good_record_clean(self):
        default = QualityThresholds()
        assert "question_too_short" not in _codes(compute_quality_signals(GOOD_RECORD, default))
