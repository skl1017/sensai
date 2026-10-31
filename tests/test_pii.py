"""Tests for the reusable PII detector (core/pii.py, issue #65)."""

from __future__ import annotations

import pytest

from core.pii import (
    PII_TYPES,
    IPIIRecognizer,
    PIIDetector,
    PIIMatch,
    RegexRecognizer,
)


@pytest.mark.parametrize(
    ("text", "pii_type", "snippet"),
    [
        ("pay with 4111 1111 1111 1111 today", "card", "4111 1111 1111 1111"),
        ("pay with 4111-1111-1111-1111 today", "card", "4111-1111-1111-1111"),
        ("pay with 4111111111111111.", "card", "4111111111111111"),
        ("FR76 3000 6000 0112 3456 7890 189", "iban", "FR76 3000 6000 0112 3456 7890 189"),
        ("iban: GB82 WEST 1234 5698 7654 32 ok", "iban", "GB82 WEST 1234 5698 7654 32"),
        ("fr76 3000 6000 0112 3456 7890 189", "iban", "fr76 3000 6000 0112 3456 7890 189"),
        ("call 06 12 34 56 78 now", "phone", "06 12 34 56 78"),
        ("call 0612345678 now", "phone", "0612345678"),
        ("call 06.12.34.56.78 now", "phone", "06.12.34.56.78"),
        ("call +33 6 12 34 56 78 now", "phone", "+33 6 12 34 56 78"),
        ("call 0033 6 12 34 56 78 now", "phone", "0033 6 12 34 56 78"),
        ("call +44 20 7946 0958 now", "phone", "+44 20 7946 0958"),
        ("write to alice@example.com please", "email", "alice@example.com"),
        (
            "write to first.last+tag@mail.example.co.uk.",
            "email",
            "first.last+tag@mail.example.co.uk",
        ),
    ],
)
def test_detects_each_type(text, pii_type, snippet):
    matches = PIIDetector().find(text)
    assert [m.type for m in matches] == [pii_type]
    assert text[matches[0].start : matches[0].end] == snippet


@pytest.mark.parametrize(
    "text",
    [
        "card 4111 1111 1111 1112",  # Luhn-invalid
        "iban FR76 3000 6000 0112 3456 7890 180",  # bad mod-97 checksum
        "iban GB82 WEST 1234 5698 7654 33",  # bad mod-97 checksum
        "in 2024 we shipped",
        "due 31/10/2026",
        "on 2026-10-31",
        "total 1 234,56 €",
        "code 12345",
        "short 0612",
        "version 3.12.1",
        "no personal data here",
    ],
)
def test_false_positives_are_not_reported(text):
    assert PIIDetector().find(text) == []


def test_empty_text():
    detector = PIIDetector()
    assert detector.find("") == []
    assert detector.mask("") == ("", [])


def test_mask_replaces_with_type_tags_and_counts():
    text = "Mail alice@example.com or 06 12 34 56 78, card 4111 1111 1111 1111."
    masked, counts = PIIDetector().mask(text)
    assert masked == "Mail [EMAIL] or [PHONE], card [CARD]."
    assert counts == [
        {"type": "email", "count": 1},
        {"type": "phone", "count": 1},
        {"type": "card", "count": 1},
    ]


def test_mask_counts_repeated_types_in_first_appearance_order():
    text = "bob@example.com, 06 12 34 56 78, carol@example.org"
    masked, counts = PIIDetector().mask(text)
    assert masked == "[EMAIL], [PHONE], [EMAIL]"
    assert counts == [{"type": "email", "count": 2}, {"type": "phone", "count": 1}]


def test_mask_without_pii_returns_text_unchanged():
    assert PIIDetector().mask("Bonjour tout le monde") == ("Bonjour tout le monde", [])


def test_iban_is_masked_as_iban():
    masked, counts = PIIDetector().mask("IBAN FR76 3000 6000 0112 3456 7890 189 merci")
    assert masked == "IBAN [IBAN] merci"
    assert counts == [{"type": "iban", "count": 1}]


def test_card_number_is_not_also_reported_as_phone():
    matches = PIIDetector().find("card 4111 1111 1111 1111")
    assert [m.type for m in matches] == ["card"]


def test_card_digits_inside_valid_iban_are_not_a_card():
    matches = PIIDetector().find("FR76 3000 6000 0112 3456 7890 189")
    assert [m.type for m in matches] == ["iban"]


def test_priority_card_beats_overlapping_custom_type():
    class OverlappingRecognizer:
        def find(self, text: str) -> list[PIIMatch]:
            return [PIIMatch("person", 5, 15)]

    detector = PIIDetector(recognizers=[OverlappingRecognizer()])
    matches = detector.find("card 4111 1111 1111 1111")
    assert [m.type for m in matches] == ["card"]


def test_same_priority_longer_span_wins():
    class LongRecognizer:
        def find(self, text: str) -> list[PIIMatch]:
            return [PIIMatch("person", 0, 12)]

    class ShortRecognizer:
        def find(self, text: str) -> list[PIIMatch]:
            return [PIIMatch("person", 0, 5)]

    matches = PIIDetector(recognizers=[ShortRecognizer(), LongRecognizer()]).find(
        "Alice Dupont habite"
    )
    assert matches == [PIIMatch("person", 0, 12)]


def test_restricting_types_disables_others():
    text = "alice@example.com 06 12 34 56 78 4111 1111 1111 1111"
    assert [m.type for m in PIIDetector(types=("email",)).find(text)] == ["email"]
    assert [m.type for m in PIIDetector(types=("card",)).find(text)] == ["card"]
    assert [m.type for m in RegexRecognizer(types=("phone",)).find(text)] == ["phone"]


def test_default_types_cover_all_builtin_types():
    text = "alice@example.com 06 12 34 56 78 FR76 3000 6000 0112 3456 7890 189 4111 1111 1111 1111"
    found = {m.type for m in PIIDetector().find(text)}
    assert found == set(PII_TYPES)


@pytest.mark.parametrize("types", [("ssn",), ("email", "passport")])
def test_unknown_type_raises_value_error(types):
    with pytest.raises(ValueError):
        RegexRecognizer(types=types)
    with pytest.raises(ValueError):
        PIIDetector(types=types)


class PersonRecognizer:
    """Stand-in for an entity-recognition model."""

    def find(self, text: str) -> list[PIIMatch]:
        index = text.find("Alice")
        if index < 0:
            return []
        return [PIIMatch("person", index, index + len("Alice"))]


def test_custom_recognizer_is_an_ipii_recognizer():
    assert isinstance(PersonRecognizer(), IPIIRecognizer)


def test_custom_recognizer_is_masked_with_its_type():
    detector = PIIDetector(recognizers=[PersonRecognizer()])
    masked, counts = detector.mask("Alice writes to bob@example.com")
    assert masked == "[PERSON] writes to [EMAIL]"
    assert counts == [{"type": "person", "count": 1}, {"type": "email", "count": 1}]


def test_repr_of_match_does_not_leak_value():
    text = "reach alice@example.com"
    (match,) = PIIDetector().find(text)
    assert not hasattr(match, "value")
    assert "alice" not in repr(match)
    assert "example.com" not in repr(match)
    assert (match.start, match.end) == (6, 23)
