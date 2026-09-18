# WATER polygon split

Source file: `init_dataset/water.csv` (386,814 ENVI pixel rows).
Script: `create_polygone_water_urban_road.py`.

Lat range: 31.35059145 to 31.42369429 (expected 31.35059145 to 31.42369429)
Lon range: 73.42515779 to 73.50928382 (expected 73.42515779 to 73.50928382)

## How the 33 polygons were split

All 33 water polygons were exported together in one CSV. Coordinates sit in a tight box, so sorting by `Lat`/`Lon` mixes pixels from different shapes.

Split method:

1. Read the combined CSV and strip header spaces so columns resolve as `Lat` and `Lon`.
2. Standardize `Lat` and `Lon` so neither axis dominates distance.
3. Run **KMeans** (`n_clusters=33`, `random_state=42`, `n_init=10`) on those coordinates.
4. Write each cluster to its own CSV. The cluster ID is not kept in the output.
5. Copy whole polygons into train / validate / test (~70 / 15 / 15 by polygon count, balanced by pixels).

Each output file has the original columns: `File X`, `File Y`, `Map X`, `Map Y`, `Lat`, `Lon`, `B1`, `B2`, `B3`, `B4`.

## Pixel count per polygon

| File | Pixels |
|---|---|
| `Polygon_water_1.csv` | 12,951 |
| `Polygon_water_2.csv` | 10,206 |
| `Polygon_water_3.csv` | 8,833 |
| `Polygon_water_4.csv` | 13,205 |
| `Polygon_water_5.csv` | 13,808 |
| `Polygon_water_6.csv` | 13,666 |
| `Polygon_water_7.csv` | 9,774 |
| `Polygon_water_8.csv` | 12,757 |
| `Polygon_water_9.csv` | 12,778 |
| `Polygon_water_10.csv` | 10,509 |
| `Polygon_water_11.csv` | 14,216 |
| `Polygon_water_12.csv` | 10,205 |
| `Polygon_water_13.csv` | 11,854 |
| `Polygon_water_14.csv` | 9,582 |
| `Polygon_water_15.csv` | 9,693 |
| `Polygon_water_16.csv` | 14,201 |
| `Polygon_water_17.csv` | 12,275 |
| `Polygon_water_18.csv` | 8,546 |
| `Polygon_water_19.csv` | 13,140 |
| `Polygon_water_20.csv` | 8,356 |
| `Polygon_water_21.csv` | 15,155 |
| `Polygon_water_22.csv` | 13,162 |
| `Polygon_water_23.csv` | 11,037 |
| `Polygon_water_24.csv` | 11,650 |
| `Polygon_water_25.csv` | 10,723 |
| `Polygon_water_26.csv` | 12,445 |
| `Polygon_water_27.csv` | 12,005 |
| `Polygon_water_28.csv` | 11,091 |
| `Polygon_water_29.csv` | 15,077 |
| `Polygon_water_30.csv` | 9,852 |
| `Polygon_water_31.csv` | 9,027 |
| `Polygon_water_32.csv` | 11,599 |
| `Polygon_water_33.csv` | 13,436 |
| **Total** | **386,814** |

## Train / validate / test

Whole polygons only (no random pixel split).

| Set | Polygons | Pixels | Share |
|---|---|---|---|
| `water_poly/train/` | 1, 2, 3, 5, 6, 7, 8, 9, 12, 13, 15, 16, 18, 20, 21, 22, 23, 24, 26, 28, 30, 32, 33 | 267,055 | 69.0% |
| `water_poly/validate/` | 10, 19, 27, 29, 31 | 59,758 | 15.4% |
| `water_poly/test/` | 4, 11, 14, 17, 25 | 60,001 | 15.5% |
| **Total** | **33** | **386,814** | **100%** |
