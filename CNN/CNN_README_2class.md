# 1D CNN: CPF 253 vs SL 284 sugarcane varieties

This folder trains a **1D convolutional network** to tell two sugarcane varieties apart from a 4-band pixel spectrum.

- **CPF 253** = class 0
- **SL 284** = class 1
- **Features:** `B1, B2, B3, B4` only (lat/lon are not used, so location cannot leak the class)
- **Split unit:** whole field polygons, not random pixels
- **Device (this run):** `cuda`
- **Best checkpoint:** epoch 2 (lowest validation loss = 0.6779)

Re-run `python CNN/run_cnn.py` to train again. This file is **rewritten automatically** at the end of that script.

## Headline results

| Split | Accuracy | Balanced acc | Kappa | ROC-AUC | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Validate | 0.7111 | 0.6784 | 0.3628 | 0.7795 | 0.6812 | 0.7084 |
| Test (once) | 0.8745 | 0.8437 | 0.7157 | 0.9520 | 0.8572 | 0.8714 |

**How to read this.** Validate is the honest check on unseen fields used during training. Test is scored **once** after the best weights are locked. Test can look higher than validate because those polygons happen to be easier, not because the test set leaked into training.

Train accuracy (~0.89) vs validate (~0.71) is the field-shift gap: the model fits training fields well, then drops on other fields. That is expected with a polygon split.

## Why this setup

ENVI exports dumped many field polygons into one CSV per variety. Neighbour pixels in the same field look almost the same. A random pixel train/test split would put the same field on both sides and inflate accuracy.

So:

1. KMeans on lat/lon made 15 CPF 253 polygons and 17 SL 284 polygons.
2. Whole polygons were copied into `train` / `validate` / `test` (~70 / 15 / 15 by pixel count).
3. SL 284 was **not** cut down to match CPF 253. Extra SL 284 pixels are more samples of a different class. Class weights handle the 1.75:1 imbalance.
4. 2D patch CNN is later work. This model is spectral only.

CPF 253 polygons: train 1,2,3,6,8,9,11,12,13,14,15 · validate 4,5 · test 7,10.

SL 284 polygons: train 2,3,6,7,8,9,10,11,12,13,15,17 · validate 1,5,16 · test 4,14.

## Data used by the CNN

| Split | CPF 253 pixels | SL 284 pixels | Total | Share |
|---|---:|---:|---:|---:|
| Train | 93,549 | 162,026 | 255,575 | 69.9% |
| Validate | 19,951 | 35,396 | 55,347 | 15.1% |
| Test | 19,530 | 35,087 | 54,617 | 14.9% |
| **All** | **133,030** | **232,509** | **365,539** | **100%** |

`StandardScaler` is fit on **train** pixels only, then applied to validate and test.

## Model

Small on purpose: four bands do not justify a large net.

- Input shape (PyTorch): `(batch, 1, 4)` — one channel, four-band sequence
- Conv1D 32, kernel 2, BatchNorm, ReLU
- Conv1D 64, kernel 2, ReLU
- Flatten → Dense 64 → Dropout 0.3 → 2-class logits
- Loss: weighted cross-entropy (CPF 253 up-weighted)
- Optimizer: Adam (`lr=1e-3`, `weight_decay=1e-4`)
- Batch size 256, max 40 epochs, early stop on validation loss (patience 8), seed 42

## How to run

GPU PyTorch (RTX 4060):

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
python CNN/run_cnn.py
```

Scripts: `config.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `plots.py`, `run_cnn.py`.

## Training set vs validation set

Early stop at epoch 10; **best validation loss at epoch 2**. Weights from that epoch are saved.

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 0.2705 | 0.8793 | 0.7160 | 0.6930 |
| 2 (best) | 0.2429 | 0.8894 | 0.6779 | 0.7111 |
| 3 | 0.2394 | 0.8901 | 0.7492 | 0.6998 |
| 4 | 0.2382 | 0.8902 | 0.7515 | 0.7016 |
| 5 | 0.2375 | 0.8901 | 0.7724 | 0.6994 |
| 6 | 0.2368 | 0.8909 | 0.7102 | 0.7089 |
| 7 | 0.2359 | 0.8913 | 0.7393 | 0.7014 |
| 8 | 0.2353 | 0.8917 | 0.7211 | 0.7065 |
| 9 | 0.2353 | 0.8911 | 0.7214 | 0.7078 |
| 10 | 0.2348 | 0.8916 | 0.7110 | 0.7165 |

![Training and validation loss](results/loss_curve.png)

![Training and validation accuracy](results/accuracy_curve.png)

## Validate

Used every epoch for early stopping. Not the final paper number.

Accuracy 0.7111. Balanced accuracy 0.6784. Cohen's kappa 0.3628. ROC-AUC 0.7795. Macro F1 0.6812. Weighted F1 0.7084.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| CPF 253 | 0.6075 | 0.5612 | 0.5834 | 19,951 |
| SL 284 | 0.7628 | 0.7956 | 0.7789 | 35,396 |
| **Overall accuracy** |  |  | **0.7111** | **55,347** |

```
              precision    recall  f1-score   support

         CPF 253     0.6075    0.5612    0.5834     19951
          SL 284     0.7628    0.7956    0.7789     35396

    accuracy                         0.7111     55347
   macro avg     0.6852    0.6784    0.6812     55347
weighted avg     0.7068    0.7111    0.7084     55347
```

|  | Predicted CPF 253 | Predicted SL 284 |
|---|---:|---:|
| Actual CPF 253 | 11,196 | 8,755 |
| Actual SL 284 | 7,234 | 28,162 |

![Validate confusion matrix](results/confusion_matrix_validate.png)

![Validate ROC](results/roc_validate.png)

![Validate precision-recall](results/pr_validate.png)

## Test (scored once)

Held-out polygons. Scored after training finished.

Accuracy 0.8745. Balanced accuracy 0.8437. Cohen's kappa 0.7157. ROC-AUC 0.9520. Macro F1 0.8572. Weighted F1 0.8714.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| CPF 253 | 0.8949 | 0.7355 | 0.8074 | 19,530 |
| SL 284 | 0.8660 | 0.9519 | 0.9070 | 35,087 |
| **Overall accuracy** |  |  | **0.8745** | **54,617** |

```
              precision    recall  f1-score   support

         CPF 253     0.8949    0.7355    0.8074     19530
          SL 284     0.8660    0.9519    0.9070     35087

    accuracy                         0.8745     54617
   macro avg     0.8805    0.8437    0.8572     54617
weighted avg     0.8764    0.8745    0.8714     54617
```

|  | Predicted CPF 253 | Predicted SL 284 |
|---|---:|---:|
| Actual CPF 253 | 14,364 | 5,166 |
| Actual SL 284 | 1,687 | 33,400 |

![Test confusion matrix](results/confusion_matrix_test.png)

![Test ROC](results/roc_test.png)

![Test precision-recall](results/pr_test.png)

## Permutation importance

Each band is shuffled on 20,000 validation pixels (4 bands × 10 repeats). The number is the **drop in accuracy** when that band is destroyed. Larger drop = the model used that band more. **B4** is the strongest band in this run.

| Band | Mean accuracy drop | Std |
|---|---:|---:|
| B4 | 0.1381 | 0.0030 |
| B1 | 0.0868 | 0.0014 |
| B2 | 0.0557 | 0.0015 |
| B3 | 0.0402 | 0.0023 |

![Permutation importance](results/permutation_importance.png)

## Files written by a run

| Path | What |
|---|---|
| `results/best_model.pt` | Weights at best validation loss |
| `results/metrics.json` | All numeric metrics |
| `results/loss_curve.png` | Train vs validate loss |
| `results/accuracy_curve.png` | Train vs validate accuracy |
| `results/confusion_matrix_validate.png` | Validate confusion matrix |
| `results/confusion_matrix_test.png` | Test confusion matrix |
| `results/roc_validate.png` / `results/roc_test.png` | ROC + AUC |
| `results/pr_validate.png` / `results/pr_test.png` | Precision-recall |
| `results/permutation_importance.png` | Band importance |
| `CNN_README.md` | This report (auto-updated) |
