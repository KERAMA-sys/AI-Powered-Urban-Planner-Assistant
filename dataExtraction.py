import os
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

# Input and output folders
input_dir = "clipped_cities_nlcd"
output_dir = "geojson_cities_nlcd"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".tif"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".tif", ".geojson"))

        with rasterio.open(input_path) as src:
            image = src.read(1)
            mask = image != src.nodata

            results = (
                {"properties": {"land_cover": int(val)}, "geometry": geom}
                for geom, val in shapes(image, mask=mask, transform=src.transform)
            )

            # Convert and clean geometry
            cleaned_features = []
            for feature in results:
                try:
                    geom = shape(feature["geometry"])
                    if geom.is_valid and not geom.is_empty and geom.area > 0:
                        feature["geometry"] = geom
                        cleaned_features.append(feature)
                except Exception as e:
                    continue  # skip broken geometry

            if not cleaned_features:
                print(f"⚠️ No valid features for {filename}")
                continue

            gdf = gpd.GeoDataFrame.from_features(cleaned_features, crs=src.crs)
            gdf = gdf.to_crs(epsg=4326)  # Save in WGS84
            gdf.to_file(output_path, driver="GeoJSON")
            print(f"✅ Saved cleaned GeoJSON: {output_path}")
