# CPF polygon split

Source file: `cpf_corrected.csv` (133,030 ENVI pixel rows).  
Script: `create_polygone_cpf.py`.

## How the 15 polygons were split

All 15 field polygons were exported together in one CSV. Their latitudes sit in a tight range (~31.383–31.405), so sorting by `Lat`/`Lon` mixes pixels from different fields.

Split method:

1. Read the combined CSV and strip header spaces so columns resolve as `Lat` and `Lon`.
2. Standardize `Lat` and `Lon` so neither axis dominates distance.
3. Run **KMeans** (`n_clusters=15`, `random_state=42`, `n_init=10`) on those coordinates.
4. Write each cluster to its own CSV. The cluster ID is not kept in the output.

Each output file has the original columns: `File X`, `File Y`, `Map X`, `Map Y`, `Lat`, `Lon`, `B1`, `B2`, `B3`, `B4`.

## Pixel count per polygon

| File | Pixels |
|---|---|
| `Polygon_cpf_1.csv` | 13,352 |
| `Polygon_cpf_2.csv` | 7,345 |
| `Polygon_cpf_3.csv` | 8,966 |
| `Polygon_cpf_4.csv` | 11,147 |
| `Polygon_cpf_5.csv` | 8,804 |
| `Polygon_cpf_6.csv` | 8,691 |
| `Polygon_cpf_7.csv` | 9,396 |
| `Polygon_cpf_8.csv` | 7,833 |
| `Polygon_cpf_9.csv` | 6,977 |
| `Polygon_cpf_10.csv` | 10,134 |
| `Polygon_cpf_11.csv` | 10,627 |
| `Polygon_cpf_12.csv` | 9,403 |
| `Polygon_cpf_13.csv` | 7,008 |
| `Polygon_cpf_14.csv` | 6,312 |
| `Polygon_cpf_15.csv` | 7,035 |
| **Total** | **133,030** |

## Train / validate / test

Whole polygons only (no random pixel split). Script: `split_train_val_test.py`.

| Set | Polygons | Pixels | Share |
|---|---|---|---|
| `cpf_poly/train/` | 1, 2, 3, 6, 8, 9, 11, 12, 13, 14, 15 | 93,549 | 70.3% |
| `cpf_poly/validate/` | 4, 5 | 19,951 | 15.0% |
| `cpf_poly/test/` | 7, 10 | 19,530 | 14.7% |
| **Total** | **15** | **133,030** | **100%** |
