"""Reusable PII (personal data) detector.

Shared by `InputGuardrailNode` (refuse or mask before the model sees the
text), `OutputGuardrailNode` (mask before the answer is cached or shown) and
the export path. The detector only reports *where* PII is and *what kind* it
is; it never keeps, logs or returns the matched values.

Types
-----
- `email`: local@domain.tld.
- `phone`: French numbers (`0X XX XX XX XX`, `+33 X ...`, `0033 ...`) and
  international E.164 numbers (`+<country><digits>`, 8 to 15 digits).
  Bare digit runs such as years, dates, prices or short numbers do not match.
- `iban`: 2 letters, 2 digits, 11 to 30 alphanumerics, spaces allowed. Only
  accepted when the ISO 13616 mod-97 checksum is valid.
- `card`: 13 to 19 digits, optionally separated by single spaces or dashes.
  Only accepted when the Luhn checksum is valid. Digits inside a valid IBAN
  are never reported as a card.

Priority rule
-------------
Matches from all recognizers are sorted by start offset. When two matches
overlap, the higher-priority type wins: card > iban > email > phone > any
other (custom) type. Within the same priority, the longer span wins. Only
kept matches are returned or masked, so a card number is never also reported
as a phone number.

Extensibility
-------------
`PIIDetector(recognizers=...)` accepts any object implementing
`IPIIRecognizer` (a `find(text) -> list[PIIMatch]` method). This is the
extension point for an entity-recognition (NER) model that finds names,
addresses or other free-form entities that regular expressions cannot
validate. Custom types are masked as `[TYPE]`, e.g. `[PERSON]`.

Masking
-------
`PIIDetector.mask(text)` replaces each kept match with `[TYPE]`
(`[EMAIL]`, `[PHONE]`, `[IBAN]`, `[CARD]`) and returns the masked text with
per-type counts ordered by first appearance.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

PII_TYPES: tuple[str, ...] = ("email", "phone", "iban", "card")

# Lower value = higher priority. Custom types default to the last rank.
_PRIORITY: dict[str, int] = {"card": 0, "iban": 1, "email": 2, "phone": 3}
_CUSTOM_PRIORITY = 4


@dataclass(frozen=True)
class PIIMatch:
    """A detected span: its type and offsets in the text. The value is not stored."""

    type: str
    start: int
    end: int


@runtime_checkable
class IPIIRecognizer(Protocol):
    """Extension point: anything that finds PII spans in a text (regex, NER model...)."""

    def find(self, text: str) -> list[PIIMatch]: ...


_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?![A-Za-z0-9-])"
)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:"
    r"0[1-9](?:[ .-]?\d{2}){4}"  # French national: 0X XX XX XX XX
    r"|(?:\+|00)33[ .-]?[1-9](?:[ .-]?\d{2}){4}"  # French international: +33 / 0033
    r"|\+[1-9](?:[ .-]?\d){7,14}"  # E.164: 8 to 15 digits in total
    r")(?!\d)"
)
# Bounded to 13..19 digits so that long digit runs are never scanned quadratically.
_CARD_RE = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
_IBAN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2}\d{2}(?:[ ]?[A-Za-z0-9]){11,30}")
_IBAN_SHAPE_RE = re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}")


def _is_alnum(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _iban_ok(candidate: str) -> bool:
    compact = candidate.replace(" ", "").upper()
    if _IBAN_SHAPE_RE.fullmatch(compact) is None:
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(char, 36)) for char in rearranged)
    return int(numeric) % 97 == 1


def _find_ibans(text: str) -> list[PIIMatch]:
    found: list[PIIMatch] = []
    pos = 0
    while (match := _IBAN_RE.search(text, pos)) is not None:
        end = _longest_valid_iban_end(text, match.start(), match.end())
        if end is None:
            pos = match.start() + 1
        else:
            found.append(PIIMatch("iban", match.start(), end))
            pos = end
    return found


def _longest_valid_iban_end(text: str, start: int, limit: int) -> int | None:
    # The greedy match may swallow trailing words ("... 189 and"): try the
    # longest prefix that ends on a token boundary and passes the checksum.
    for end in range(limit, start, -1):
        if not _is_alnum(text[end - 1]):
            continue
        if end < len(text) and _is_alnum(text[end]):
            continue
        if _iban_ok(text[start:end]):
            return end
    return None


def _find_cards(text: str, ibans: list[PIIMatch]) -> list[PIIMatch]:
    found: list[PIIMatch] = []
    pos = 0
    while (match := _CARD_RE.search(text, pos)) is not None:
        overlaps_iban = any(match.start() < iban.end and iban.start < match.end() for iban in ibans)
        digits = re.sub(r"\D", "", match.group())
        if not overlaps_iban and 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append(PIIMatch("card", match.start(), match.end()))
            pos = match.end()
        else:
            pos = match.start() + 1
    return found


class RegexRecognizer:
    """Finds email, phone, IBAN and card PII with regular expressions and checksums."""

    def __init__(self, types: Iterable[str] = PII_TYPES) -> None:
        wanted = frozenset(types)
        unknown = sorted(wanted.difference(PII_TYPES))
        if unknown:
            raise ValueError(
                f"unknown PII type(s): {', '.join(unknown)} (expected among {', '.join(PII_TYPES)})"
            )
        self._types = wanted

    def find(self, text: str) -> list[PIIMatch]:
        found: list[PIIMatch] = []
        # IBANs are always computed: card digits inside a valid IBAN are excluded.
        ibans = _find_ibans(text)
        if "email" in self._types:
            found += [PIIMatch("email", m.start(), m.end()) for m in _EMAIL_RE.finditer(text)]
        if "phone" in self._types:
            found += [PIIMatch("phone", m.start(), m.end()) for m in _PHONE_RE.finditer(text)]
        if "iban" in self._types:
            found += ibans
        if "card" in self._types:
            found += _find_cards(text, ibans)
        return sorted(found, key=lambda m: (m.start, m.end))


class PIIDetector:
    """Combines the regex recognizer with extra recognizers, resolves overlaps, masks."""

    def __init__(
        self,
        types: Iterable[str] = PII_TYPES,
        recognizers: Iterable[IPIIRecognizer] = (),
    ) -> None:
        self._recognizers: list[IPIIRecognizer] = [RegexRecognizer(types), *recognizers]

    def find(self, text: str) -> list[PIIMatch]:
        candidates = [match for recognizer in self._recognizers for match in recognizer.find(text)]
        candidates.sort(
            key=lambda m: (_PRIORITY.get(m.type, _CUSTOM_PRIORITY), -(m.end - m.start), m.start)
        )
        kept: list[PIIMatch] = []
        for candidate in candidates:
            if not any(candidate.start < k.end and k.start < candidate.end for k in kept):
                kept.append(candidate)
        return sorted(kept, key=lambda m: (m.start, m.end))

    def mask(self, text: str) -> tuple[str, list[dict]]:
        matches = self.find(text)
        if not matches:
            return text, []
        parts: list[str] = []
        counts: dict[str, int] = {}
        cursor = 0
        for match in matches:
            parts.append(text[cursor : match.start])
            parts.append(f"[{match.type.upper()}]")
            cursor = match.end
            counts[match.type] = counts.get(match.type, 0) + 1
        parts.append(text[cursor:])
        return "".join(parts), [{"type": t, "count": n} for t, n in counts.items()]
