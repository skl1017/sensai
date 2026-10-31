# Semantic cache threshold bench (issue #61)

Calibrates the cosine threshold (default 0.92) above which a new question is served a
cached answer. Labelled set: 49 question pairs (24 `same: true` paraphrases, 25 `same: false`
hard negatives), mostly French, some English.

## Setup

- Embedder: `nomic-embed-text` via Ollama (`http://localhost:11434`), through `OllamaEmbedder`,
  raw text, no task prefix.
- Pairs: `eval/cache_pairs.yaml`. Scores: cosine (`nodes/fewshot_section.py::_cosine`).
- Sweep: thresholds 0.80 to 0.99, step 0.01. A pair is a hit when `score >= threshold`.

## Reproduce

```bash
.venv/bin/python -m eval.cache_threshold                     # full sweep, scores, recommendation
.venv/bin/pytest -q tests/test_cache_threshold.py            # unit tests, no Ollama
```

## Results

Subset of the sweep around the interesting zone (FP = different pair served the cached answer,
FN = same pair missed, i.e. an extra LLM call):

| threshold | FP | FN | precision | recall |
|---:|---:|---:|---:|---:|
| 0.88 | 16 | 3 | 0.568 | 0.875 |
| 0.90 | 10 | 3 | 0.677 | 0.875 |
| 0.91 | 9 | 3 | 0.700 | 0.875 |
| **0.92** (current default) | 9 | 4 | 0.690 | 0.833 |
| 0.93 | 7 | 4 | 0.741 | 0.833 |
| 0.94 | 4 | 7 | 0.810 | 0.708 |
| 0.95 | 2 | 9 | 0.882 | 0.625 |
| **0.96** (recommended) | 1 | 13 | 0.917 | 0.458 |
| 0.97 | 1 | 16 | 0.889 | 0.333 |
| 0.98 | 1 | 20 | 0.800 | 0.167 |
| 0.99 | 1 | 23 | 0.500 | 0.042 |

No threshold in the range reaches zero false positives.

## Overlap

- Lowest score among `same: true` pairs: **0.6900** ("How do I reset my password?" / "how can I reset my pwd").
- Highest score among `same: false` pairs: **0.9902** ("Est-ce que je dois réviser avant l'examen ?" / "Est-ce que je ne dois pas réviser avant l'examen ?").
- The two classes overlap across the whole 0.69 to 0.99 band: the embedder does not separate negations or single-digit or single-word changes.
- Worst hard negatives (score >= 0.92, all served wrongly at 0.92):

| score | different pair |
|---:|---|
| 0.9902 | devoir réviser / ne pas devoir réviser (negation) |
| 0.9573 | boucle **for** / boucle **while** en Python |
| 0.9426 | joueurs équipe de **football** / de **rugby** |
| 0.9406 | ouvrir / **fermer** un compte bancaire |
| 0.9392 | distance **Terre-Lune** / **Terre-Soleil** |
| 0.9382 | installer python sur **windows** / sur **linux** |
| 0.9372 | combien font **2+2** / **2+3** |
| 0.9212 | tour Eiffel en **mètres** / en **pieds** |
| 0.9210 | Révolution **française** / **russe** (année) |

## Chosen threshold

**0.96**, the value `recommend()` returns. Because no threshold reaches zero FP, the fallback
rule applies: the lowest FP is 1 (every threshold from 0.96 to 0.99), and among those the fewest
FN is at 0.96 (13 FN, recall 0.46).

Why not 0.92: it lets 9 different questions through, and the default is worse than the
recommendation on every axis except recall (0.83 vs 0.46). Why not higher than 0.96: from 0.97 on,
recall drops below 0.34 for the same single FP, so it only adds misses.

0.96 is still not safe on its own. The remaining FP is the negation pair (0.9902, above any
usable threshold), so the threshold cannot carry the guarantee. Before enabling the cache, it
needs a guard that refuses a hit when the two questions disagree on a discriminating token:
negation (`ne`, `pas`, `not`, `n'`), numbers (would have blocked 2+2/2+3 and 2010/2020), or units.
Until that guard exists, keep the cache disabled.

## Caveats

- Measured with raw text (no `search_query:` / `clustering:` prefix that `nomic-embed-text`
  documents). That matches how `OllamaEmbedder` is used today. A prefix may shift the whole score
  distribution, so the threshold is only valid for this exact setup.
- 49 pairs, hand-written: the FP and FN counts are small-sample. One pair moves FP by one.
- Re-measure whenever the embedder model or its options change (`eval.cache_threshold --model ...`).
  The table is only valid for `nomic-embed-text` as of this run.
