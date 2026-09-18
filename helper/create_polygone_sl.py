import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# 1. Load your ENVI output file
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "sl_corrected.csv")

print("Reading massive ENVI CSV file...")
# skipinitialspace handles "File X, File Y, Lat, Lon" style headers
df = pd.read_csv(input_file, comment=";", skipinitialspace=True)
df.columns = df.columns.str.strip()

print(f"Loaded {len(df):,} pixels")
print(f"Columns: {list(df.columns)}")
print(f"Lat range: {df['Lat'].min():.8f} to {df['Lat'].max():.8f}")
print(f"Lon range: {df['Lon'].min():.8f} to {df['Lon'].max():.8f}")

# 2. Automatically find the 17 distinct spatial polygon shapes
print("Grouping pixels into 17 independent polygons...")
kmeans = KMeans(n_clusters=17, random_state=42, n_init=10)

# Scale Lat/Lon so neither axis dominates clustering
coords = StandardScaler().fit_transform(df[["Lat", "Lon"]])
df["Polygon_ID"] = kmeans.fit_predict(coords)

# 3. Export each polygon to its own clean CSV file
output_dir = os.path.join(script_dir, "sl_poly")
os.makedirs(output_dir, exist_ok=True)

print("Exporting individual CSV files...")
for i in range(17):
    polygon_df = df[df["Polygon_ID"] == i].drop(columns=["Polygon_ID"])
    output_name = os.path.join(output_dir, f"Polygon_sl_{i + 1}.csv")
    polygon_df.to_csv(output_name, index=False)
    print(f"saved: sl_poly/Polygon_sl_{i + 1}.csv ({len(polygon_df)} pixels)")

print("\nSuccess! All 17 individual polygon CSV files have been created.")
