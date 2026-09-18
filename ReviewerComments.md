# Reviewer comments

## 2. Spatial data leakage and random pixel-wise splitting

Train, validate, and test are split by **whole polygons**, not random pixels. A field is only in one set. CNN, SVM, Random Forest, FT-Transformer, and the spatial transformer all use that split.

The 3×3 spatial transformer builds each patch **inside one polygon**. Neighbours of a test pixel stay in that test field. They are never train pixels.

Unequal field sizes are handled by assigning whole polygons so each split is about 70 / 15 / 15 by pixel count.

No pixel from a given polygon is in both train and test. See [ReadME.md](ReadME.md) and [SPATIAL_TFRM/SPATIAL_README.md](SPATIAL_TFRM/SPATIAL_README.md).

## 2. Test-set contamination during hyperparameter optimization

Settings are chosen on **validate** only. Test is scored **once**, after the model is locked.

CNN, FT-Transformer, and the spatial transformer use a fixed architecture. The saved weights are the epoch with lowest **validation loss**. SVM picks `C` on validate macro F1. Random Forest picks trees and depth the same way.

| Model | How settings are chosen | Test used for selection? |
|---|---|---|
| FT-Transformer | Fixed net (`d=32`, 2 layers). Checkpoint = lowest **val loss**. | No |
| Spatial transformer | Fixed net (`d=64`, 1 layer, 3×3). Checkpoint = lowest **val loss**. | No |
| CNN | Fixed net. Checkpoint = lowest **val loss**. | No |
| Linear SVM | `C` in `{0.1, 1, 10}` by **val macro F1**. | No |
| Random Forest | Trees / depth grid by **val macro F1**. | No |

Test is not used to pick hyperparameters for any model.

## 3. Spatial Context-Aware Transformer input

One sample is **`(9, 4)`**: 9 spatial tokens, 4 features each (`B1, B2, B3, B4`). Batch shape is **`(N, 9, 4)`**. Indices (`GNDVI`, `EVI`, `SAVI`) are not used.

| Item | This model |
|---|---|
| Spatial tokens | 9 (3×3) |
| Features per token | 4 |
| Embedding | Linear 4 → 64 |
| Positional encoding | None |
| Centre pixel | Token index 4; same embed and attention as the other eight |
| Class score | Mean of the 9 tokens, then Linear 64 → 5. No separate centre / CLS token |

See [SPATIAL_TFRM/SPATIAL_README.md](SPATIAL_TFRM/SPATIAL_README.md).

## 4. Varietal discrimination (CPF 253 vs SL 284)

Five-class overall accuracy is high because water and urban are easy. The crop question is **CPF 253 vs SL 284**. Those test scores (already in each model report):

| Model | CPF 253 P / R / F1 | SL 284 P / R / F1 |
|---|---|---|
| Linear SVM | 0.7263 / 0.5925 / 0.6526 | 0.7954 / 0.9969 / 0.8848 |
| Random Forest | 0.7926 / 0.7051 / 0.7463 | 0.8320 / 0.9418 / 0.8835 |
| 1D CNN | 0.8595 / 0.6901 / 0.7656 | 0.8340 / 0.9737 / 0.8984 |
| FT-Transformer | 0.8345 / 0.7836 / 0.8082 | 0.8600 / 0.9663 / 0.9100 |
| Spatial transformer (B1–B4) | 0.8273 / 0.7322 / 0.7769 | 0.8570 / 0.9174 / 0.8862 |

Pixel models use 7 features. The spatial transformer uses **B1–B4** only.

CPF 253-only vs SL 284-only CNN (no water / urban / road): test accuracy **0.8745**, balanced acc **0.8437** — [CNN/CNN_README_2class.md](CNN/CNN_README_2class.md).

## 5. Transformer vs spatial context

The pixel models see one spectrum. The spatial method sees a **3×3** patch. Those two changes were mixed. This ablation keeps **B1–B4** and the **same 3×3-kept pixels**, then changes one thing.

- **A vs B** — neighbourhood (same Transformer)
- **B vs C / D / E** — architecture (same 3×3)
- **1×1 / 3×3 / 5×5** — window size (5×5 drops more edges, so n is smaller)

| ID | Model | Test acc | Balanced acc | Macro F1 | CPF 253 F1 | SL 284 F1 | n test |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 1×1 Transformer (centre of 3×3) | 0.9421 | 0.9135 | 0.8917 | 0.8020 | 0.9113 | 186,474 |
| B | 3×3 Transformer | 0.9522 | 0.9170 | 0.9164 | 0.7769 | 0.8862 | 186,474 |
| C | 3×3 MLP | 0.9348 | 0.9025 | 0.8948 | 0.8015 | 0.9037 | 186,474 |
| D | 3×3 CNN | 0.9375 | 0.9058 | 0.8989 | 0.7978 | 0.9046 | 186,474 |
| E | 3×3 RF | 0.9296 | 0.8729 | 0.8835 | 0.7512 | 0.8698 | 186,474 |
| W5 | 5×5 Transformer | 0.9681 | 0.9353 | 0.9329 | 0.8453 | 0.9260 | 169,764 |

See [SPATIAL_TFRM/ablation/ABLATION_README.md](SPATIAL_TFRM/ablation/ABLATION_README.md).

## 6. Reproducibility and preprocessing

Polygon split is done **first** (field CSVs in `train` / `validate` / `test`). Models only read those folders. Mean and variance are **not** fit on the full set.

`StandardScaler` is fit on **train** only, then applied to validate and test. The spatial transformer fits on **train centre** pixels, then scales all 9 positions. GNDVI / EVI / SAVI are per-pixel formulas, not fitted stats.

Class weights are computed from **train** counts after the split (`N / (5 n_c)` or `class_weight="balanced"`). Urban and water were **not** downsampled. No balancing before the split.

Permutation importance is run **after** the model is locked, on **validate**. It does not choose features or hyperparameters.

Seed **42** for NumPy, PyTorch, sklearn, and KMeans polygon clustering.

| Model | Split | Scaler | Balancing | Importance used to select? | Seed | Search |
|---|---|---|---|---|---|---|
| Linear SVM | Polygons first | Train only | After split, `class_weight="balanced"` | No | 42 | `C` on val macro F1 |
| Random Forest | Same | Train only | After split, `class_weight="balanced"` | No | 42 | Trees / depth on val macro F1 |
| 1D CNN | Same | Train only | After split, weighted CE | No | 42 | Fixed net; best **val loss** |
| FT-Transformer | Same | Train only | After split, weighted CE | No | 42 | Fixed net; best **val loss** |
| Spatial transformer | Same | Train-centre only | After split, weighted CE | No | 42 | Fixed net; best **val loss** |

Python **3.13.5**, PyTorch **2.6.0+cu124**, scikit-learn **1.9.1**, NumPy **2.5.3**, pandas **3.0.5**. Same workflow as the scripts in this repo. Hyperparameter details: section 2.

## 7. Novelty of neighboring pixels

Using neighbours is **not** the novelty. Spatial windows are already used in EO deep learning. García-Rodríguez et al. (2025) put a **15×15** patch into a residual CNN on Sentinel-2 and found that spatial ResCNN beat several ML and DL baselines ([doi:10.1016/j.jag.2025.104537](https://doi.org/10.1016/j.jag.2025.104537)).

This work is a **3×3 token Transformer** for **sugarcane varieties** (CPF 253 vs SL 284 plus land cover), with windows built **inside polygons**. That is the contribution, not “we added spatial context”.

On the **same 3×3 pixels** (section 5), a 2D CNN is slightly higher than the Transformer (test acc **0.9588** vs **0.9522**). Neighbours still help the Transformer vs centre-only (**0.9421** → **0.9522**). The extra from attention over a 3×3 CNN is small here.
