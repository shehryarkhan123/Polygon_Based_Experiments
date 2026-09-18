# SL polygon split

Source file: `sl_corrected.csv` (232,509 ENVI pixel rows).  
Script: `create_polygone_sl.py`.

Lat range: 31.38680774 to 31.40613698  
Lon range: 73.44392032 to 73.47701715

## How the 17 polygons were split

All 17 field polygons were exported together in one CSV. Coordinates sit in a tight box, so sorting by `Lat`/`Lon` mixes pixels from different fields.

Split method:

1. Read the combined CSV and strip header spaces so columns resolve as `Lat` and `Lon`.
2. Standardize `Lat` and `Lon` so neither axis dominates distance.
3. Run **KMeans** (`n_clusters=17`, `random_state=42`, `n_init=10`) on those coordinates.
4. Write each cluster to its own CSV. The cluster ID is not kept in the output.

Each output file has the original columns: `File X`, `File Y`, `Map X`, `Map Y`, `Lat`, `Lon`, `B1`, `B2`, `B3`, `B4`.

Pixel counts were **not** reduced to match CPF (133,030). SL is a different class; extra pixels are kept. Class imbalance can be handled later with class weights.

## Pixel count per polygon

| File | Pixels |
|---|---|
| `Polygon_sl_1.csv` | 9,206 |
| `Polygon_sl_2.csv` | 14,947 |
| `Polygon_sl_3.csv` | 15,583 |
| `Polygon_sl_4.csv` | 18,909 |
| `Polygon_sl_5.csv` | 13,396 |
| `Polygon_sl_6.csv` | 8,577 |
| `Polygon_sl_7.csv` | 14,618 |
| `Polygon_sl_8.csv` | 15,614 |
| `Polygon_sl_9.csv` | 14,895 |
| `Polygon_sl_10.csv` | 18,091 |
| `Polygon_sl_11.csv` | 7,280 |
| `Polygon_sl_12.csv` | 14,630 |
| `Polygon_sl_13.csv` | 14,152 |
| `Polygon_sl_14.csv` | 16,178 |
| `Polygon_sl_15.csv` | 14,225 |
| `Polygon_sl_16.csv` | 12,794 |
| `Polygon_sl_17.csv` | 9,414 |
| **Total** | **232,509** |

## Train / validate / test

Whole polygons only (no random pixel split). Script: `split_train_val_test.py`.

| Set | Polygons | Pixels | Share |
|---|---|---|---|
| `sl_poly/train/` | 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17 | 162,026 | 69.7% |
| `sl_poly/validate/` | 1, 5, 16 | 35,396 | 15.2% |
| `sl_poly/test/` | 4, 14 | 35,087 | 15.1% |
| **Total** | **17** | **232,509** | **100%** |
