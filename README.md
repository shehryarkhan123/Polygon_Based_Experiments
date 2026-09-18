#  POLYGON BASED SUGARCANE VARIETIES CLASSIFICATION
###### CLASSIFICATION & ANALYSIS
Polygon-based crop classification: split combined ENVI pixel CSVs (CPF 253 and SL 284) into individual field polygons with KMeans on lat/lon.

## Docs

| Readme | What it covers |
|---|---|
| [CPF 253 polygons](cpf_readme.md) | 15 CPF 253 fields, pixel counts, train / validate / test split |
| [SL 284 polygons](sl_readme.md) | 17 SL 284 fields, pixel counts, train / validate / test split |
| [Water polygons](water_readme.md) | 33 water polygons from `init_dataset/water.csv` |
| [Urban polygons](urban_readme.md) | 45 urban polygons from `init_dataset/urban.csv` |
| [Road polygons](road_readme.md) | 12 road polygons from `init_dataset/road.csv` |
| [1D CNN 5-class](CNN/CNN_README.md) | CPF 253, SL 284, water, urban, road; 7 features; train/validate/test |
| [1D CNN 2-class archive](CNN/CNN_README_2class.md) | Original CPF 253 vs SL 284 (4-band) results |
| [Linear SVM 5-class](SVM/SVM_README.md) | Same splits and 7 features; LinearSVC |
| [Random Forest 5-class](RANDOM_FOREST/RF_README.md) | Same splits and 7 features; RandomForest |
| [FT-Transformer 5-class](TABULAR_TFRM/TFR_README.md) | Same splits and 7 features; small tabular transformer |
| [Spatial Transformer 5-class](SPATIAL_TFRM/SPATIAL_README.md) | Same polygon splits; 3×3 neighbors × 4 features (`B1`–`B4`) |
| [Spatial ablation](SPATIAL_TFRM/ablation/ABLATION_README.md) | 1×1 vs 3×3 Transformer; 3×3 MLP / CNN / RF; 5×5 window |
| [Conclusion](conclusion.md) | All models, per-class test P/R/F1 and short conclusions |
| [Reviewer comments](ReviewerComments.md) | Polygon split; train-only scaler; spatial windows not claimed as novel |

## Train / validate / test split

**Why.** The model must be checked on fields it has never seen. Random pixel split leaks neighbors from the same field into train and test, so accuracy looks high but is fake.

**How.** Whole polygons only. CPF 253/SL 284: `split_train_val_test.py`. Water/urban/road: `create_polygone_water_urban_road.py`. About 70% train, 15% validate, 15% test by pixel count.

- CPF 253: 11 / 2 / 2 polygons → `cpf_poly/train`, `cpf_poly/validate`, `cpf_poly/test` — [cpf_readme.md](cpf_readme.md)
- SL 284: 12 / 3 / 2 polygons → `sl_poly/train`, `sl_poly/validate`, `sl_poly/test` — [sl_readme.md](sl_readme.md)
- Water: 23 / 5 / 5 polygons → `water_poly/` — [water_readme.md](water_readme.md)
- Urban: 31 / 7 / 7 polygons → `urban_poly/` — [urban_readme.md](urban_readme.md)
- Road: 8 / 2 / 2 polygons → `road_poly/` — [road_readme.md](road_readme.md)

**When.** After polygons are created, before CNN training. Validate is used while tuning. Test is used once at the end.

## 1D CNN

Train 5 classes (CPF 253, SL 284, water, urban, road) on B1–B4 + GNDVI, EVI, SAVI with GPU PyTorch:

```
python CNN/run_cnn.py
```

Full 5-class report: [CNN/CNN_README.md](CNN/CNN_README.md). Archived 2-class CPF 253 vs SL 284: [CNN/CNN_README_2class.md](CNN/CNN_README_2class.md).

## Linear SVM

Same 5 classes and 7 features, full train set, `class_weight="balanced"`:

```
python SVM/run_svm.py
```

Report: [SVM/SVM_README.md](SVM/SVM_README.md).

## Random Forest

Same 5 classes and 7 features, full train set, `class_weight="balanced"`:

```
python RANDOM_FOREST/run_rf.py
```

Report: [RANDOM_FOREST/RF_README.md](RANDOM_FOREST/RF_README.md).

## FT-Transformer

Same 5 classes and 7 features, full train set, inverse-frequency class weights:

```
python TABULAR_TFRM/run_tfr.py
```

Report: [TABULAR_TFRM/TFR_README.md](TABULAR_TFRM/TFR_README.md).

## Spatial-context Transformer

Same polygon splits. Each sample is a 3×3 patch (9 pixels × 4 features `B1`–`B4`), inverse-frequency class weights:

```
python SPATIAL_TFRM/run_spatial.py
```

Report: [SPATIAL_TFRM/SPATIAL_README.md](SPATIAL_TFRM/SPATIAL_README.md). Ablation (Transformer vs spatial context): `python SPATIAL_TFRM/ablation/run_ablation.py` — [SPATIAL_TFRM/ablation/ABLATION_README.md](SPATIAL_TFRM/ablation/ABLATION_README.md).

Per-class comparison of every model: [conclusion.md](conclusion.md).
