# Benchmarks locaux des modèles

Généré par `python -m eval.bench`. Médianes sur les runs mesurés (p95 entre parenthèses).

## Machines

| Machine | CPU | Cœurs | RAM (Go) | GPU | Ollama | Date |
|---|---|---|---|---|---|---|
| DESKTOP-V6CATDH | AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD | 16 | — | NVIDIA GeForce RTX 5080, 16303 MiB | 0.40.11 | 2026-10-27 |
| sch-Vivobook-Go-E1404FA-E1404FA | AMD Ryzen 5 7520U with Radeon Graphics | 8 | 16.0 | — | 0.34.4 | 2026-10-30 |
| zen | AMD Ryzen 7 7735U with Radeon Graphics | 16 | 16.0 | — | 0.34.4 | 2026-10-27 |

## Résultats

| Machine | Modèle | Prompt | Load (s) | TTFT (ms) | Latence (ms) | Gen tok/s | Prompt tok/s | Taille (Go) | VRAM (Go) | % GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| DESKTOP-V6CATDH | qwen2.5:3b | short | 16.1 | 24 (28) | 410 (476) | 255 (257) | 2121 (2477) | 2.4 | 2.4 | 100 |
| DESKTOP-V6CATDH | qwen2.5:3b | long | 16.1 | 203 (248) | 711 (772) | 244 (245) | 15248 (15510) | 2.4 | 2.4 | 100 |
| DESKTOP-V6CATDH | qwen2.5:3b | tool | 16.1 | 150 (211) | 160 (219) | 247 (254) | 8930 (9497) | 2.4 | 2.4 | 100 |
| sch-Vivobook-Go-E1404FA-E1404FA | qwen2.5:3b | short | 4.6 | 552 (557) | 5722 (6125) | 15 (15) | 53 (54) | 2.4 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | qwen2.5:3b | long | 4.6 | 53923 (54011) | 63905 (64912) | 12 (12) | 42 (42) | 2.4 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | qwen2.5:3b | tool | 4.6 | 6065 (6108) | 6204 (6244) | 15 (15) | 48 (48) | 2.4 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | llama3.2:3b | short | 4.4 | 556 (561) | 6131 (6146) | 15 (15) | 76 (76) | 3.1 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | llama3.2:3b | long | 4.4 | 56339 (56402) | 75082 (75136) | 7 (7) | 41 (41) | 3.1 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | llama3.2:3b | tool | 4.4 | 5682 (5754) | 5750 (5823) | 15 (15) | 51 (51) | 3.1 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | qwen3:4b | short | 5.1 | 5841 (6765) | 11425 (11524) | 12 (12) | 41 (42) | 3.9 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | qwen3:4b | long | 5.1 | — | 106376 (107480) | 4 (4) | 29 (29) | 3.9 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | qwen3:4b | tool | 5.1 | — | 17863 (18130) | 10 (11) | 36 (36) | 3.9 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | qwen2.5:7b | short | 9.4 | 1392 (1977) | 10411 (11282) | 7 (7) | 21 (21) | 5.5 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | qwen2.5:7b | long | 9.4 | 119586 (120498) | 136031 (136226) | 6 (6) | 19 (19) | 5.5 | 0.0 | 0 |
| sch-Vivobook-Go-E1404FA-E1404FA | qwen2.5:7b | tool | 9.4 | 13534 (13724) | 13821 (14013) | 7 (7) | 21 (21) | 5.5 | 0.0 | 0 |
| zen | qwen2.5:3b | short | 4.6 | 268 (316) | 4610 (5225) | 17 (18) | 111 (124) | 2.4 | 0.0 | 0 |
| zen | qwen2.5:3b | long | 4.6 | 25827 (26551) | 34696 (35460) | 14 (15) | 88 (94) | 2.4 | 0.0 | 0 |
| zen | qwen2.5:3b | tool | 4.6 | 3621 (3847) | 3736 (3962) | 18 (18) | 101 (102) | 2.4 | 0.0 | 0 |
