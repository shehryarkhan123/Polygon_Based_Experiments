# Spatial ablation: Transformer vs spatial context (B1–B4)

Same polygon split as the spatial transformer. Features are **B1, B2, B3, B4** only.
A–E use the **same 3×3-kept pixels**. The 5×5 Transformer uses its own kept set (more edge drop).

Device: `cuda`. Test scored **once**. Settings on validate only.

Re-run `python SPATIAL_TFRM/ablation/run_ablation.py`.

## Same 3×3 pixels

| ID | Model | Test acc | Balanced acc | Macro F1 | CPF 253 F1 | SL 284 F1 | n test |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 1×1 Transformer (centre of 3×3) | 0.9421 | 0.9135 | 0.8917 | 0.8020 | 0.9113 | 186,474 |
| B | 3×3 Transformer | 0.9522 | 0.9170 | 0.9164 | 0.7769 | 0.8862 | 186,474 |
| C | 3×3 MLP | 0.9348 | 0.9025 | 0.8948 | 0.8015 | 0.9037 | 186,474 |
| D | 3×3 CNN | 0.9475 | 0.9158 | 0.8989 | 0.7978 | 0.9046 | 186,474 |
| E | 3×3 RF | 0.9296 | 0.8729 | 0.8835 | 0.7512 | 0.8698 | 186,474 |

A vs B: neighbourhood (same Transformer). B vs C/D/E: architecture (same 3×3).

## Window size

| ID | Model | Test acc | Balanced acc | Macro F1 | CPF 253 F1 | SL 284 F1 | n test |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 1x1 Transformer (centre of 3x3) | 0.9421 | 0.9135 | 0.8917 | 0.8020 | 0.9113 | 186,474 |
| B | 3x3 Transformer | 0.9522 | 0.9170 | 0.9164 | 0.7769 | 0.8862 | 186,474 |
| W5 | 5x5 Transformer | 0.9681 | 0.9353 | 0.9329 | 0.8453 | 0.9260 | 169,764 |

