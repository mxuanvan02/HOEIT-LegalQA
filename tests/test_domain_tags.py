"""Tests for legal-domain tag inference.

Domain tags drive the per-domain breakdown in the paper (Bảng B.1), so the
identifier -> domain mapping must stay stable and the placeholder values must
never leak into a reported table.
"""

from __future__ import annotations

from domain_tags import (
    DOMAIN_MAP,
    infer_domain_tag_from_identifier,
    infer_domain_tag_from_record,
)


class TestInferFromIdentifier:
    def test_known_prefix_maps_to_domain(self):
        assert infer_domain_tag_from_identifier("13") == "Civil Law I"

    def test_identifier_with_suffix_uses_leading_number(self):
        assert infer_domain_tag_from_identifier("13_abc") == "Civil Law I"

    def test_leading_whitespace_tolerated(self):
        assert infer_domain_tag_from_identifier("  24 ") == "Civil Procedure"

    def test_unknown_prefix_returns_other(self):
        assert infer_domain_tag_from_identifier("99") == "Other"

    def test_non_numeric_returns_unknown(self):
        assert infer_domain_tag_from_identifier("chapter-1") == "unknown"

    def test_empty_returns_unknown(self):
        assert infer_domain_tag_from_identifier("") == "unknown"


class TestInferFromRecord:
    def test_existing_tag_wins(self):
        record = {"domain_tag": "Labor Law", "doc_id": "13"}
        assert infer_domain_tag_from_record(record) == "Labor Law"

    def test_placeholder_tag_is_replaced_by_inference(self):
        record = {"domain_tag": "unknown", "doc_id": "38"}
        assert infer_domain_tag_from_record(record) == "Labor Law"

    def test_blank_tag_is_replaced_by_inference(self):
        record = {"domain_tag": "   ", "chunk_id": "25"}
        assert infer_domain_tag_from_record(record) == "Family Law"

    def test_doc_id_is_tried_before_chunk_id(self):
        record = {"doc_id": "13", "chunk_id": "25"}
        assert infer_domain_tag_from_record(record) == "Civil Law I"

    def test_falls_back_to_unknown_when_nothing_resolves(self):
        assert infer_domain_tag_from_record({}) == "unknown"

    def test_placeholder_civil_law_is_not_returned(self):
        record = {"domain_tag": "civil_law", "doc_id": "13"}
        assert infer_domain_tag_from_record(record) == "Civil Law I"


class TestDomainMapIntegrity:
    def test_all_values_are_non_empty(self):
        assert all(isinstance(v, str) and v.strip() for v in DOMAIN_MAP.values())

    def test_all_keys_are_numeric_strings(self):
        assert all(k.isdigit() for k in DOMAIN_MAP)

    def test_map_is_not_empty(self):
        assert len(DOMAIN_MAP) > 0
