"""Unit tests for the cache threshold calibration (`eval/cache_threshold.py`).

No Ollama: scores come from handcrafted values, or from `FakeEmbedder`.
"""

from __future__ import annotations

import pytest

from eval.cache_threshold import load_pairs, recommend, score_pairs, sweep
from tests.fakes import FakeEmbedder


def _write(tmp_path, text: str):
    path = tmp_path / "pairs.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_pairs_reads_valid_file(tmp_path):
    path = _write(
        tmp_path,
        "pairs:\n"
        '  - {a: "Quelle est la capitale de la France ?", '
        'b: "capitale de la France ?", same: true}\n'
        '  - {a: "combien font 2+2 ?", b: "combien font 2+3 ?", same: false}\n',
    )

    pairs = load_pairs(path)

    assert pairs == [
        {
            "a": "Quelle est la capitale de la France ?",
            "b": "capitale de la France ?",
            "same": True,
        },
        {"a": "combien font 2+2 ?", "b": "combien font 2+3 ?", "same": False},
    ]


def test_load_pairs_rejects_missing_pairs_key(tmp_path):
    with pytest.raises(ValueError, match="'pairs' list"):
        load_pairs(_write(tmp_path, "other: []\n"))


def test_load_pairs_rejects_non_boolean_same(tmp_path):
    path = _write(tmp_path, 'pairs:\n  - {a: "x", b: "y", same: "yes"}\n')
    with pytest.raises(ValueError, match="boolean 'same'"):
        load_pairs(path)


def test_load_pairs_rejects_empty_question(tmp_path):
    path = _write(tmp_path, 'pairs:\n  - {a: "  ", b: "y", same: true}\n')
    with pytest.raises(ValueError, match="'a' and 'b'"):
        load_pairs(path)


def test_load_pairs_reads_shipped_file():
    pairs = load_pairs("eval/cache_pairs.yaml")
    assert len(pairs) >= 40
    assert any(p["same"] for p in pairs)
    assert any(not p["same"] for p in pairs)


def test_sweep_counts_fp_and_fn_per_threshold():
    scores = [0.95, 0.90, 0.85, 0.70]
    labels = [True, False, True, False]

    rows = sweep(scores, labels, [0.80, 0.91, 0.92])

    assert rows[0] == {"threshold": 0.80, "fp": 1, "fn": 0, "precision": 2 / 3, "recall": 1.0}
    assert rows[1] == {"threshold": 0.91, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5}
    assert rows[2] == {"threshold": 0.92, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5}


def test_sweep_is_inclusive_at_threshold():
    rows = sweep([0.92], [True], [0.92])
    assert rows[0]["fn"] == 0
    assert rows[0]["recall"] == 1.0


def test_sweep_conventions_when_nothing_predicted_or_no_positive():
    rows = sweep([0.1], [False], [0.9])
    assert rows[0]["precision"] == 1.0
    assert rows[0]["recall"] == 0.0


def test_sweep_rejects_length_mismatch():
    with pytest.raises(ValueError):
        sweep([0.5, 0.6], [True], [0.9])


def test_recommend_picks_lowest_zero_fp_threshold():
    rows = sweep([0.95, 0.90, 0.85, 0.70], [True, False, True, False], [0.80, 0.88, 0.91, 0.92])
    # 0.80 and 0.88 admit the 0.90 false positive; 0.91 is the lowest zero-FP one.
    assert recommend(rows)["threshold"] == 0.91


def test_recommend_breaks_ties_on_fewest_fn():
    rows = [
        {"threshold": 0.90, "fp": 0, "fn": 3, "precision": 1.0, "recall": 0.5},
        {"threshold": 0.90, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.8},
    ]
    assert recommend(rows)["fn"] == 1


def test_recommend_without_zero_fp_minimises_fp_then_fn():
    rows = [
        {"threshold": 0.90, "fp": 2, "fn": 0, "precision": 0.5, "recall": 1.0},
        {"threshold": 0.95, "fp": 1, "fn": 4, "precision": 0.5, "recall": 0.5},
        {"threshold": 0.97, "fp": 1, "fn": 2, "precision": 0.5, "recall": 0.7},
    ]
    assert recommend(rows)["threshold"] == 0.97


def test_recommend_rejects_empty_sweep():
    with pytest.raises(ValueError):
        recommend([])


async def test_score_pairs_identical_questions_score_one():
    pairs = [{"a": "quelle heure est il", "b": "quelle heure est il", "same": True}]
    scores = await score_pairs(FakeEmbedder(), pairs)
    assert scores == [pytest.approx(1.0)]


async def test_score_pairs_orders_by_word_overlap():
    embedder = FakeEmbedder()
    pairs = [
        {"a": "capitale de la France", "b": "capitale de la France", "same": True},
        {"a": "capitale de la France", "b": "capitale de la France ou pas", "same": True},
    ]

    identical, partial = await score_pairs(embedder, pairs)

    assert identical > partial > 0.0


async def test_score_pairs_embeds_in_one_batch():
    embedder = FakeEmbedder()
    pairs = [
        {"a": "un", "b": "deux", "same": False},
        {"a": "trois", "b": "quatre", "same": False},
    ]

    scores = await score_pairs(embedder, pairs)

    assert len(scores) == 2
    assert embedder.calls == [["un", "deux", "trois", "quatre"]]


async def test_score_pairs_empty_input():
    embedder = FakeEmbedder()
    assert await score_pairs(embedder, []) == []
    assert embedder.calls == []
