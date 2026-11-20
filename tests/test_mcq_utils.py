"""Tests for MCQ candidate normalization and ground-truth resolution.

These helpers decide whether a synthetic answer is scored as correct, so a
silent change here would move published benchmark numbers. The cases below
pin the resolution order (exact -> payload -> label -> fuzzy).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "research"))

from mcq_utils import (  # noqa: E402
    build_ground_truth_variants,
    canonicalize_text,
    extract_choice_label,
    extract_embedded_ground_truth,
    normalize_text,
    resolve_ground_truth_index,
    strip_choice_label,
    strip_trailing_notes,
)

CANDIDATES = [
    "A. Viện kiểm sát phê chuẩn lệnh bắt.",
    "B. Tòa án nhân dân phê chuẩn lệnh bắt.",
    "C. Cơ quan điều tra tự phê chuẩn lệnh bắt.",
    "D. Ủy ban nhân dân cấp xã phê chuẩn lệnh bắt.",
]


class TestNormalize:
    def test_collapses_whitespace_and_case(self):
        assert normalize_text("  Viện   KIỂM sát \n") == "viện kiểm sát"

    def test_non_string_is_coerced(self):
        assert normalize_text(None) == "none"


class TestChoiceLabel:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("A. Viện kiểm sát.", "A"),
            ("b) Tòa án", "B"),
            ("C: Cơ quan điều tra", "C"),
            ("D - Ủy ban", "D"),
        ],
    )
    def test_extracts_labels(self, text, expected):
        assert extract_choice_label(text) == expected

    def test_returns_none_without_label(self):
        assert extract_choice_label("Viện kiểm sát phê chuẩn") is None

    def test_label_outside_a_to_d_is_not_matched(self):
        assert extract_choice_label("E. Ngoài phạm vi") is None

    def test_strip_removes_only_leading_label(self):
        assert strip_choice_label("A. Viện kiểm sát.") == "Viện kiểm sát."


class TestTrailingNotes:
    def test_strips_english_scaffolding_note(self):
        assert strip_trailing_notes("Viện kiểm sát (full correct answer text)") == "Viện kiểm sát"

    def test_strips_vietnamese_note(self):
        assert strip_trailing_notes("Viện kiểm sát (đây là đáp án đúng)") == "Viện kiểm sát"

    def test_keeps_text_without_note(self):
        assert strip_trailing_notes("Viện kiểm sát") == "Viện kiểm sát"


class TestEmbeddedGroundTruth:
    def test_extracts_last_tag(self):
        text = "<ground_truth>A. Viện kiểm sát</ground_truth>"
        assert extract_embedded_ground_truth(text) == "A. Viện kiểm sát"

    def test_returns_plain_text_unchanged(self):
        assert extract_embedded_ground_truth("A. Viện kiểm sát") == "A. Viện kiểm sát"


class TestCanonicalize:
    def test_strips_label_note_and_case(self):
        assert canonicalize_text("A. Viện kiểm sát (full correct answer text)") == "viện kiểm sát"

    def test_variants_are_deduplicated_and_ordered(self):
        variants = build_ground_truth_variants("A. Viện kiểm sát")
        assert variants[0] == "A. Viện kiểm sát"
        assert len(variants) == len(set(variants))


class TestResolveGroundTruth:
    def test_exact_full_match_wins(self):
        res = resolve_ground_truth_index(CANDIDATES, "A. Viện kiểm sát phê chuẩn lệnh bắt.")
        assert res["matched_index"] == 0
        assert res["method"] == "exact_full"

    def test_payload_match_when_only_the_note_differs(self):
        # The note is stripped, so this is an exact payload match.
        res = resolve_ground_truth_index(
            CANDIDATES, "Viện kiểm sát phê chuẩn lệnh bắt. (full correct answer text)"
        )
        assert res["matched_index"] == 0
        assert res["method"] == "payload_match"

    def test_trailing_punctuation_difference_falls_back_to_fuzzy(self):
        # canonicalize_text does NOT strip sentence punctuation, so a ground
        # truth that drops the final "." is not a payload match and falls
        # through to the fuzzy path. The index is still correct.
        res = resolve_ground_truth_index(
            CANDIDATES, "Viện kiểm sát phê chuẩn lệnh bắt (full correct answer text)"
        )
        assert res["matched_index"] == 0
        assert res["method"] == "fuzzy_payload"

    def test_label_only_ground_truth(self):
        # Label-only matching requires the payload to be empty after the label
        # is stripped, i.e. "A." rather than a bare "A".
        res = resolve_ground_truth_index(CANDIDATES, "A.")
        assert res["matched_index"] == 0
        assert res["method"] == "label_only"

    def test_bare_label_is_not_treated_as_label_only(self):
        # "A" alone has a non-empty payload ("a"), so it does not reach the
        # label-only branch and stays unresolved rather than guessing.
        res = resolve_ground_truth_index(CANDIDATES, "A")
        assert res["matched_index"] is None
        assert res["method"] == "unresolved"

    def test_unresolvable_ground_truth(self):
        res = resolve_ground_truth_index(CANDIDATES, "Một đáp án không tồn tại trong danh sách")
        assert res["matched_index"] is None
        assert res["method"] == "unresolved"

    def test_fuzzy_disabled_returns_unresolved(self):
        res = resolve_ground_truth_index(
            CANDIDATES,
            "Viện kiểm sat phe chuan lenh bat",
            allow_fuzzy=False,
        )
        assert res["method"] != "fuzzy_payload"

    def test_empty_candidates(self):
        res = resolve_ground_truth_index([], "A. Viện kiểm sát")
        assert res["method"] == "no_candidates"
        assert res["matched_index"] is None

    def test_non_list_candidates(self):
        res = resolve_ground_truth_index("not a list", "A. Viện kiểm sát")
        assert res["method"] == "candidates_not_list"
        assert res["candidate_count"] == 0

    def test_matched_index_is_deterministic_across_repeats(self):
        first = resolve_ground_truth_index(CANDIDATES, "C. Cơ quan điều tra tự phê chuẩn lệnh bắt.")
        for _ in range(5):
            assert resolve_ground_truth_index(
                CANDIDATES, "C. Cơ quan điều tra tự phê chuẩn lệnh bắt."
            ) == first
        assert first["matched_index"] == 2
