# Conclusion: 5-class test comparison

Same polygon split for every model. Test scored **once**. Pixel models (SVM, RF, 1D CNN, FT-Transformer) use **7 features** on all test pixels (**203,690**). The spatial transformer uses **B1–B4** only and drops pixels without a full 3×3 inside the polygon (**186,474**). Those two groups are not a like-for-like overall-accuracy contest.

Full reports: [SVM](SVM/SVM_README.md), [RF](RANDOM_FOREST/RF_README.md), [1D CNN](CNN/CNN_README.md), [FT-Transformer](TABULAR_TFRM/TFR_README.md), [spatial](SPATIAL_TFRM/SPATIAL_README.md). Ablation: [ABLATION_README.md](SPATIAL_TFRM/ablation/ABLATION_README.md).

## Overall (test)

| Model | Features | Test acc | Balanced acc | Macro F1 |
|---|---|---:|---:|---:|
| Linear SVM | 7 | 0.9053 | 0.8262 | 0.8310 |
| Random Forest | 7 | 0.9251 | 0.8752 | 0.8723 |
| 1D CNN | 7 | 0.9298 | 0.8888 | 0.8804 |
| FT-Transformer | 7 | 0.9338 | 0.8921 | 0.8876 |
| Spatial transformer | 4, 3×3 | 0.9522 | 0.9170 | 0.9164 |

Among **7-feature pixel** models, FT-Transformer is highest on balanced acc and macro F1. Spatial is higher still, but it sees neighbours and a smaller pixel set.

## Test F1 by class

| Class | SVM | RF | 1D CNN | FT-Transformer | Spatial 3×3 |
|---|---:|---:|---:|---:|---:|
| CPF 253 | 0.6526 | 0.7463 | 0.7656 | **0.8082** | 0.7769 |
| SL 284 | 0.8848 | 0.8835 | 0.8984 | **0.9100** | 0.8862 |
| water | 0.9799 | 0.9827 | 0.9841 | 0.9787 | 0.9986 |
| urban | 0.9488 | 0.9674 | 0.9662 | 0.9682 | 0.9916 |
| road | 0.6886 | 0.7813 | 0.7879 | 0.7730 | **0.9289** |

## Test P / R / F1

### CPF 253

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Linear SVM | 0.7263 | 0.5925 | 0.6526 |
| Random Forest | 0.7926 | 0.7051 | 0.7463 |
| 1D CNN | 0.8595 | 0.6901 | 0.7656 |
| FT-Transformer | 0.8345 | 0.7836 | 0.8082 |
| Spatial transformer | 0.8273 | 0.7322 | 0.7769 |

Hardest class. Mistakes go mainly to SL 284 (unseen fields). Best pixel F1: **FT-Transformer**. Spatial does not beat FT on CPF 253.

### SL 284

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Linear SVM | 0.7954 | 0.9969 | 0.8848 |
| Random Forest | 0.8320 | 0.9418 | 0.8835 |
| 1D CNN | 0.8340 | 0.9737 | 0.8984 |
| FT-Transformer | 0.8600 | 0.9663 | 0.9100 |
| Spatial transformer | 0.8570 | 0.9174 | 0.8862 |

Recall is high on every model. SVM almost never misses SL 284; that is why CPF 253 recall is low. Best F1: **FT-Transformer**.

### Water

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Linear SVM | 0.9915 | 0.9686 | 0.9799 |
| Random Forest | 0.9965 | 0.9694 | 0.9827 |
| 1D CNN | 0.9967 | 0.9717 | 0.9841 |
| FT-Transformer | 0.9985 | 0.9596 | 0.9787 |
| Spatial transformer | 0.9993 | 0.9980 | 0.9986 |

Easy for all models. Inflates 5-class accuracy. Not the crop question.

### Urban

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Linear SVM | 0.9620 | 0.9359 | 0.9488 |
| Random Forest | 0.9768 | 0.9582 | 0.9674 |
| 1D CNN | 0.9839 | 0.9491 | 0.9662 |
| FT-Transformer | 0.9771 | 0.9594 | 0.9682 |
| Spatial transformer | 0.9951 | 0.9881 | 0.9916 |

Same pattern as water: spectrally distinct, large class, all models strong.

### Road

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Linear SVM | 0.7492 | 0.6371 | 0.6886 |
| Random Forest | 0.7623 | 0.8014 | 0.7813 |
| 1D CNN | 0.7274 | 0.8592 | 0.7879 |
| FT-Transformer | 0.7552 | 0.7918 | 0.7730 |
| Spatial transformer | 0.9095 | 0.9491 | 0.9289 |

Thin class. SVM is weakest. Neighbours help: spatial 3×3 is clearly best.

## Ablation (B1–B4, same 3×3 pixels except W5)

| ID | Model | Test acc | Macro F1 | CPF 253 F1 | SL 284 F1 |
|---|---|---:|---:|---:|---:|
| A | 1×1 Transformer | 0.9421 | 0.8917 | 0.8020 | 0.9113 |
| B | 3×3 Transformer | 0.9522 | 0.9164 | 0.7769 | 0.8862 |
| C | 3×3 MLP | 0.9348 | 0.8948 | 0.8015 | 0.9037 |
| D | 3×3 CNN | 0.9375 | 0.8989 | 0.7978 | 0.9046 |
| E | 3×3 RF | 0.9296 | 0.8835 | 0.7512 | 0.8698 |
| W5 | 5×5 Transformer | 0.9681 | 0.9329 | 0.8453 | 0.9260 |

Neighbours help vs centre-only (A → B). On the same 3×3 patches, the Transformer outperforms the 2D CNN and MLP. W5 uses fewer pixels (more edge drop).

## Bottom line

- **Do not read overall accuracy as variety skill.** Water and urban are easy.
- **Crop task:** CPF 253 vs SL 284. Best 7-feature pixel model: **FT-Transformer**. CPF 253 stays the weak class.
- **Road:** spatial 3×3 is the clear gain from neighbourhood.
- **Transformer vs spatial context:** the window matters; the Transformer outperforms the 3×3 CNN and MLP on the same spatial patches.