# URBAN polygon split

Source file: `init_dataset/urban.csv` (495,553 ENVI pixel rows).
Script: `create_polygone_water_urban_road.py`.

Lat range: 31.38062559 to 31.42014691 (expected 31.38062559 to 31.42014691)
Lon range: 73.44539945 to 73.49051373 (expected 73.44539945 to 73.49051373)

## How the 45 polygons were split

All 45 urban polygons were exported together in one CSV. Coordinates sit in a tight box, so sorting by `Lat`/`Lon` mixes pixels from different shapes.

Split method:

1. Read the combined CSV and strip header spaces so columns resolve as `Lat` and `Lon`.
2. Standardize `Lat` and `Lon` so neither axis dominates distance.
3. Run **KMeans** (`n_clusters=45`, `random_state=42`, `n_init=10`) on those coordinates.
4. Write each cluster to its own CSV. The cluster ID is not kept in the output.
5. Copy whole polygons into train / validate / test (~70 / 15 / 15 by polygon count, balanced by pixels).

Each output file has the original columns: `File X`, `File Y`, `Map X`, `Map Y`, `Lat`, `Lon`, `B1`, `B2`, `B3`, `B4`.

## Pixel count per polygon

| File | Pixels |
|---|---|
| `Polygon_urban_1.csv` | 12,666 |
| `Polygon_urban_2.csv` | 11,033 |
| `Polygon_urban_3.csv` | 10,651 |
| `Polygon_urban_4.csv` | 12,280 |
| `Polygon_urban_5.csv` | 16,034 |
| `Polygon_urban_6.csv` | 10,716 |
| `Polygon_urban_7.csv` | 9,340 |
| `Polygon_urban_8.csv` | 12,680 |
| `Polygon_urban_9.csv` | 9,597 |
| `Polygon_urban_10.csv` | 12,704 |
| `Polygon_urban_11.csv` | 11,420 |
| `Polygon_urban_12.csv` | 10,508 |
| `Polygon_urban_13.csv` | 11,525 |
| `Polygon_urban_14.csv` | 10,763 |
| `Polygon_urban_15.csv` | 11,239 |
| `Polygon_urban_16.csv` | 10,706 |
| `Polygon_urban_17.csv` | 9,663 |
| `Polygon_urban_18.csv` | 8,971 |
| `Polygon_urban_19.csv` | 13,066 |
| `Polygon_urban_20.csv` | 11,933 |
| `Polygon_urban_21.csv` | 9,481 |
| `Polygon_urban_22.csv` | 11,382 |
| `Polygon_urban_23.csv` | 11,679 |
| `Polygon_urban_24.csv` | 11,555 |
| `Polygon_urban_25.csv` | 12,016 |
| `Polygon_urban_26.csv` | 10,487 |
| `Polygon_urban_27.csv` | 9,481 |
| `Polygon_urban_28.csv` | 10,405 |
| `Polygon_urban_29.csv` | 12,086 |
| `Polygon_urban_30.csv` | 12,792 |
| `Polygon_urban_31.csv` | 9,244 |
| `Polygon_urban_32.csv` | 10,859 |
| `Polygon_urban_33.csv` | 8,977 |
| `Polygon_urban_34.csv` | 10,213 |
| `Polygon_urban_35.csv` | 9,857 |
| `Polygon_urban_36.csv` | 9,862 |
| `Polygon_urban_37.csv` | 10,499 |
| `Polygon_urban_38.csv` | 10,857 |
| `Polygon_urban_39.csv` | 11,874 |
| `Polygon_urban_40.csv` | 8,907 |
| `Polygon_urban_41.csv` | 10,165 |
| `Polygon_urban_42.csv` | 10,966 |
| `Polygon_urban_43.csv` | 11,606 |
| `Polygon_urban_44.csv` | 13,037 |
| `Polygon_urban_45.csv` | 9,771 |
| **Total** | **495,553** |

## Train / validate / test

Whole polygons only (no random pixel split).

| Set | Polygons | Pixels | Share |
|---|---|---|---|
| `urban_poly/train/` | 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 30, 34, 36, 39, 40, 41, 42, 43 | 341,432 | 68.9% |
| `urban_poly/validate/` | 12, 19, 24, 29, 31, 32, 35 | 77,175 | 15.6% |
| `urban_poly/test/` | 4, 13, 33, 37, 38, 44, 45 | 76,946 | 15.5% |
| **Total** | **45** | **495,553** | **100%** |
