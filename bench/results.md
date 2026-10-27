# Benchmarks locaux des modèles

Généré par `python -m eval.bench`. Médianes sur les runs mesurés (p95 entre parenthèses).

## Machines

| Machine | CPU | Cœurs | RAM (Go) | GPU | Ollama | Date |
|---|---|---|---|---|---|---|
| sch-Vivobook-Go-E1404FA-E1404FA | AMD Ryzen 5 7520U with Radeon Graphics | 8 | 16.0 | — | 0.34.4 | 2026-10-27 |

## Résultats

| Machine | Modèle | Prompt | Load (s) | TTFT (ms) | Latence (ms) | Gen tok/s | Prompt tok/s | Taille (Go) | VRAM (Go) | % GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| sch-Vivobook-Go-E1404FA-E1404FA | qwen2.5:3b | short | 8.9 | 599 (685) | 8868 (9530) | 9 (9) | 49 (52) | 2.4 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | qwen2.5:3b | long | 8.9 | 55378 (55956) | 70807 (71477) | 8 (8) | 41 (41) | 2.4 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | qwen2.5:3b | tool | 8.9 | 7320 (7368) | 7537 (7597) | 9 (9) | 47 (48) | 2.4 | 0.0 | 0 |
