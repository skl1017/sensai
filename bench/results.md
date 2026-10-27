# Benchmarks locaux des modèles

Généré par `python -m eval.bench`. Médianes sur les runs mesurés (p95 entre parenthèses).

## Machines

| Machine | CPU | Cœurs | RAM (Go) | GPU | Ollama | Date |
|---|---|---|---|---|---|---|
| DESKTOP-V6CATDH | AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD | 16 | — | NVIDIA GeForce RTX 5080, 16303 MiB | 0.40.11 | 2026-10-27 |
| sch-Vivobook-Go-E1404FA-E1404FA | AMD Ryzen 5 7520U with Radeon Graphics | 8 | 16.0 | — | 0.34.4 | 2026-10-27 |
| zen | AMD Ryzen 7 7735U with Radeon Graphics | 16 | 16.0 | — | 0.34.4 | 2026-10-27 |

## Résultats

| Machine | Modèle | Prompt | Load (s) | TTFT (ms) | Latence (ms) | Gen tok/s | Prompt tok/s | Taille (Go) | VRAM (Go) | % GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| DESKTOP-V6CATDH | qwen2.5:3b | short | 16.1 | 24 (28) | 410 (476) | 255 (257) | 2121 (2477) | 2.4 | 2.4 | 100 |
| DESKTOP-V6CATDH | qwen2.5:3b | long | 16.1 | 203 (248) | 711 (772) | 244 (245) | 15248 (15510) | 2.4 | 2.4 | 100 |
| DESKTOP-V6CATDH | qwen2.5:3b | tool | 16.1 | 150 (211) | 160 (219) | 247 (254) | 8930 (9497) | 2.4 | 2.4 | 100 |
| sch-Vivobook-Go-E1404FA-E1404FA | qwen2.5:3b | short | 8.9 | 599 (685) | 8868 (9530) | 9 (9) | 49 (52) | 2.4 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | qwen2.5:3b | long | 8.9 | 55378 (55956) | 70807 (71477) | 8 (8) | 41 (41) | 2.4 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | qwen2.5:3b | tool | 8.9 | 7320 (7368) | 7537 (7597) | 9 (9) | 47 (48) | 2.4 | 0.0 | 0 |
| zen | qwen2.5:3b | short | 4.6 | 268 (316) | 4610 (5225) | 17 (18) | 111 (124) | 2.4 | 0.0 | 0 |
| zen | qwen2.5:3b | long | 4.6 | 25827 (26551) | 34696 (35460) | 14 (15) | 88 (94) | 2.4 | 0.0 | 0 |
| zen | qwen2.5:3b | tool | 4.6 | 3621 (3847) | 3736 (3962) | 18 (18) | 101 (102) | 2.4 | 0.0 | 0 |
