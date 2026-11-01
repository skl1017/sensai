"""Unit tests for the judge calibration (`eval/calibration.py`).

No Ollama and no `nodes/eval_node.py`: the judge is a fake `judge_fn`.
"""

from __future__ import annotations

import pytest

from eval.calibration import (
    CRITERIA,
    annotated,
    load_items,
    metrics,
    render_report,
    run,
    spearman,
)

HUMAN_NULL = {"relevance": None, "coherence": None, "faithfulness": None}


def _write(tmp_path, text: str):
    path = tmp_path / "sample.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _item(question: str, human: dict, sources: list[str] | None = None) -> dict:
    return {"question": question, "answer": "réponse", "sources": sources or [], "human": human}


def test_metrics_mae_bias_and_agreements():
    pairs = [
        (
            {"relevance": 3, "coherence": 5, "faithfulness": None},
            {"relevance": 4, "coherence": 5, "faithfulness": None},
        ),
        (
            {"relevance": 2, "coherence": 1, "faithfulness": None},
            {"relevance": 2, "coherence": 3, "faithfulness": None},
        ),
        (
            {"relevance": 5, "coherence": 4, "faithfulness": None},
            {"relevance": 2, "coherence": 4, "faithfulness": None},
        ),
    ]

    stats = metrics(pairs)["relevance"]

    # diffs judge - human: +1, 0, -3
    assert stats["n"] == 3
    assert stats["mae"] == pytest.approx(4 / 3)
    assert stats["bias"] == pytest.approx(-2 / 3)
    assert stats["exact"] == pytest.approx(1 / 3)
    assert stats["within1"] == pytest.approx(2 / 3)

    coherence = metrics(pairs)["coherence"]
    # diffs: 0, +2, 0
    assert coherence["mae"] == pytest.approx(2 / 3)
    assert coherence["bias"] == pytest.approx(2 / 3)
    assert coherence["exact"] == pytest.approx(2 / 3)
    assert coherence["within1"] == pytest.approx(2 / 3)


def test_metrics_perfect_judge_has_zero_error_and_perfect_rho():
    pairs = [
        (
            {"relevance": h, "coherence": h, "faithfulness": None},
            {"relevance": h, "coherence": h, "faithfulness": None},
        )
        for h in (0, 2, 3, 5)
    ]
    stats = metrics(pairs)["relevance"]
    assert stats["mae"] == 0
    assert stats["bias"] == 0
    assert stats["exact"] == 1.0
    assert stats["within1"] == 1.0
    assert stats["spearman"] == pytest.approx(1.0)


def test_metrics_skips_none_per_criterion():
    pairs = [
        (
            {"relevance": 4, "coherence": None, "faithfulness": None},
            {"relevance": 4, "coherence": 2, "faithfulness": 3},
        ),
        (
            {"relevance": None, "coherence": 3, "faithfulness": 1},
            {"relevance": 0, "coherence": 3, "faithfulness": 5},
        ),
    ]
    result = metrics(pairs)
    assert result["relevance"]["n"] == 1
    assert result["coherence"]["n"] == 1
    assert result["faithfulness"]["n"] == 1
    assert result["faithfulness"]["bias"] == pytest.approx(4.0)


def test_metrics_skips_failed_judge_pairs():
    pairs = [
        ({"relevance": 1, "coherence": 1, "faithfulness": None}, None),
        (
            {"relevance": 1, "coherence": 1, "faithfulness": None},
            {"relevance": 1, "coherence": 1, "faithfulness": None},
        ),
    ]
    result = metrics(pairs)
    assert result["relevance"]["n"] == 1
    assert result["relevance"]["mae"] == 0


def test_metrics_empty_criterion_gives_none():
    result = metrics([(HUMAN_NULL, {"relevance": 2, "coherence": 2, "faithfulness": 2})])
    assert result["relevance"] == {
        "n": 0,
        "mae": None,
        "bias": None,
        "exact": None,
        "within1": None,
        "spearman": None,
    }


def test_spearman_perfect_and_inverse():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_handles_ties_with_average_ranks():
    # xs ranks: [1, 2.5, 2.5, 4]; ys ranks: [1, 2, 3, 4]
    xs = [1, 2, 2, 3]
    ys = [1, 2, 3, 4]
    rx = [1.0, 2.5, 2.5, 4.0]
    ry = [1.0, 2.0, 3.0, 4.0]
    mx, my = sum(rx) / 4, sum(ry) / 4
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    expected = cov / (vx * vy) ** 0.5
    assert spearman(xs, ys) == pytest.approx(expected)
    assert 0 < spearman(xs, ys) < 1


def test_spearman_undefined_cases_return_none():
    assert spearman([3], [4]) is None
    assert spearman([], []) is None
    assert spearman([2, 2, 2], [1, 2, 3]) is None
    assert spearman([1, 2, 3], [5, 5, 5]) is None


def test_load_items_and_annotated_filter(tmp_path):
    path = _write(
        tmp_path,
        "items:\n"
        '  - question: "q1"\n'
        '    answer: "a1"\n'
        '    sources: ["s1"]\n'
        "    human: {relevance: 4, coherence: null, faithfulness: 3}\n"
        '  - question: "q2"\n'
        '    answer: "a2"\n'
        "    human: {relevance: null, coherence: null, faithfulness: null}\n",
    )

    items = load_items(path)
    assert items[0]["human"] == {"relevance": 4, "coherence": None, "faithfulness": 3}
    assert items[1]["sources"] == []

    kept = annotated(items)
    assert [item["question"] for item in kept] == ["q1"]


def test_load_items_rejects_bad_shape(tmp_path):
    with pytest.raises(ValueError, match="'items' list"):
        load_items(_write(tmp_path, "other: []\n"))

    bad_score = _write(
        tmp_path,
        'items:\n  - question: "q"\n    answer: "a"\n'
        "    human: {relevance: 7, coherence: null, faithfulness: null}\n",
    )
    with pytest.raises(ValueError, match="0-5"):
        load_items(bad_score)


def test_template_loads_and_has_only_null_human_scores():
    items = load_items("eval/calibration_sample.yaml")

    assert len(items) >= 5
    for item in items:
        assert item["question"]
        assert item["answer"]
        assert item["human"] == HUMAN_NULL
    assert annotated(items) == []


async def test_run_pairs_human_with_judge_result():
    items = [
        _item("q1", {"relevance": 4, "coherence": 4, "faithfulness": None}, ["src"]),
        _item("q2", {"relevance": 1, "coherence": 2, "faithfulness": None}),
    ]
    calls: list[tuple] = []

    async def fake_judge(llm, question, answer, sources):
        calls.append((llm, question, answer, sources))
        return {"relevance": 3, "coherence": 5, "faithfulness": None, "unsupported_claims": []}

    pairs = await run(items, "LLM", fake_judge)

    assert len(pairs) == 2
    assert pairs[0][0] == items[0]["human"]
    assert pairs[0][1]["relevance"] == 3
    assert calls[0] == ("LLM", "q1", "réponse", ["src"])
    assert len(calls) == 2


async def test_run_counts_none_and_exceptions_as_failures():
    items = [
        _item("ok", {"relevance": 2, "coherence": 2, "faithfulness": None}),
        _item("none", {"relevance": 2, "coherence": 2, "faithfulness": None}),
        _item("boom", {"relevance": 2, "coherence": 2, "faithfulness": None}),
    ]

    async def flaky_judge(llm, question, answer, sources):
        if question == "none":
            return None
        if question == "boom":
            raise RuntimeError("judge down")
        return {"relevance": 2, "coherence": 2, "faithfulness": None}

    pairs = await run(items, None, flaky_judge)

    assert [judged is None for _, judged in pairs] == [False, True, True]
    assert metrics(pairs)["relevance"]["n"] == 1


def test_render_report_contains_criteria_and_hints():
    pairs = [
        (
            {"relevance": 2, "coherence": 2, "faithfulness": None},
            {"relevance": 4, "coherence": 2, "faithfulness": None},
        ),
        (
            {"relevance": 3, "coherence": 3, "faithfulness": None},
            {"relevance": 5, "coherence": 3, "faithfulness": None},
        ),
    ]
    report = render_report(metrics(pairs), failures=1, model="qwen2.5:7b")

    for name in CRITERIA:
        assert name in report
    assert "qwen2.5:7b" in report
    assert "| critère |" in report
    assert "juge indulgent" in report
    assert "Échecs du juge" in report
    assert "| — |" in report  # faithfulness has no pair
