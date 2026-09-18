# ROAD polygon split

Source file: `init_dataset/road.csv` (69,203 ENVI pixel rows).
Script: `create_polygone_water_urban_road.py`.

Lat range: 31.34255964 to 31.40953181 (expected 31.34255964 to 31.40953181)
Lon range: 73.41053294 to 73.50314023 (expected 73.41053294 to 73.50314023)

## How the 12 polygons were split

All 12 road polygons were exported together in one CSV. Coordinates sit in a tight box, so sorting by `Lat`/`Lon` mixes pixels from different shapes.

Split method:

1. Read the combined CSV and strip header spaces so columns resolve as `Lat` and `Lon`.
2. Standardize `Lat` and `Lon` so neither axis dominates distance.
3. Run **KMeans** (`n_clusters=12`, `random_state=42`, `n_init=10`) on those coordinates.
4. Write each cluster to its own CSV. The cluster ID is not kept in the output.
5. Copy whole polygons into train / validate / test (~70 / 15 / 15 by polygon count, balanced by pixels).

Each output file has the original columns: `File X`, `File Y`, `Map X`, `Map Y`, `Lat`, `Lon`, `B1`, `B2`, `B3`, `B4`.

## Pixel count per polygon

| File | Pixels |
|---|---|
| `Polygon_road_1.csv` | 8,262 |
| `Polygon_road_2.csv` | 1,369 |
| `Polygon_road_3.csv` | 3,269 |
| `Polygon_road_4.csv` | 15,647 |
| `Polygon_road_5.csv` | 4,481 |
| `Polygon_road_6.csv` | 4,766 |
| `Polygon_road_7.csv` | 4,963 |
| `Polygon_road_8.csv` | 4,087 |
| `Polygon_road_9.csv` | 3,448 |
| `Polygon_road_10.csv` | 6,540 |
| `Polygon_road_11.csv` | 8,039 |
| `Polygon_road_12.csv` | 4,332 |
| **Total** | **69,203** |

## Train / validate / test

Whole polygons only (no random pixel split).

| Set | Polygons | Pixels | Share |
|---|---|---|---|
| `road_poly/train/` | 2, 3, 4, 5, 6, 7, 10, 12 | 45,367 | 65.6% |
| `road_poly/validate/` | 1, 9 | 11,710 | 16.9% |
| `road_poly/test/` | 8, 11 | 12,126 | 17.5% |
| **Total** | **12** | **69,203** | **100%** |
