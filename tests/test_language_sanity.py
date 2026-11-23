"""Tests for Vietnamese release sanitation and artifact detection.

These guards decide whether a synthetic record leaks foreign-language or
prompt scaffolding into the public dataset, so their boundaries are pinned
here rather than left to a live dataset scan.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from language_sanity import (  # noqa: E402
    detect_foreign_artifacts,
    qa_artifact_reasons,
    sanitize_generated_text,
)


class TestSanitize:
    def test_removes_english_scaffolding_note(self):
        assert sanitize_generated_text("Viện kiểm sát (full correct answer text)") == "Viện kiểm sát"

    def test_normalizes_crlf_to_newline(self):
        assert sanitize_generated_text("a\r\nb") == "a\nb"

    def test_collapses_blank_runs(self):
        assert sanitize_generated_text("a\n\n\n\nb") == "a\n\nb"

    def test_strips_leading_and_trailing_whitespace(self):
        assert sanitize_generated_text("   Viện kiểm sát   ") == "Viện kiểm sát"

    def test_empty_input(self):
        assert sanitize_generated_text("") == ""


class TestDetectForeignArtifacts:
    def test_clean_vietnamese_has_no_reasons(self):
        assert detect_foreign_artifacts("Viện kiểm sát phê chuẩn lệnh bắt.") == []

    def test_detects_cjk_characters(self):
        reasons = detect_foreign_artifacts("根据提供的背景信息，答案是C。")
        assert "cjk_chars" in reasons

    def test_detects_english_meta_phrase(self):
        reasons = detect_foreign_artifacts("The correct answer is B because the statute says so.")
        assert "english_meta" in reasons

    def test_detects_prompt_leak_tags(self):
        reasons = detect_foreign_artifacts("<qa_pair> câu hỏi")
        assert "prompt_leak" in reasons

    def test_detects_bloom_taxonomy_leak(self):
        assert "prompt_leak" in detect_foreign_artifacts("Apply Bloom's Taxonomy level 3")

    def test_scaffolding_note_alone_is_not_an_artifact(self):
        assert detect_foreign_artifacts("Viện kiểm sát (full correct answer text)") == []


class TestQaArtifactReasons:
    def test_clean_record_has_no_reasons(self):
        reasons = qa_artifact_reasons(
            "Theo quy định, ai phê chuẩn lệnh bắt?",
            ["A. Viện kiểm sát.", "B. Tòa án.", "C. Cơ quan điều tra.", "D. Ủy ban."],
            "A. Viện kiểm sát.",
            "Tiền đề: theo luật. Kết luận: Viện kiểm sát phê chuẩn.",
        )
        assert reasons == []

    def test_artifact_in_candidate_is_detected(self):
        reasons = qa_artifact_reasons(
            "Câu hỏi tiếng Việt",
            ["A. Đáp án một.", "B. 根据法律规定，答案是B。", "C. Đáp án ba.", "D. Đáp án bốn."],
            "A. Đáp án một.",
            "Tiền đề: theo luật. Kết luận: đáp án một.",
        )
        assert "cjk_chars" in reasons

    def test_reasons_are_deduplicated(self):
        reasons = qa_artifact_reasons(
            "根据根据",
            ["A. 根据答案。", "B. B.", "C. C.", "D. D."],
            "B. B.",
            "根据根据",
        )
        assert len(reasons) == len(set(reasons))
