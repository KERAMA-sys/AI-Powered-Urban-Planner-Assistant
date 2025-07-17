import os
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from tqdm import tqdm

input_dir = "clipped_cities_nlcd"
output_dir = "geojson_cities_nlcd"
os.makedirs(output_dir, exist_ok=True)

# Loop through all TIF files
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith(".tif"):
        tif_path = os.path.join(input_dir, filename)
        geojson_path = os.path.join(output_dir, filename.replace(".tif", ".geojson"))

        with rasterio.open(tif_path) as src:
            image = src.read(1)
            mask = image != src.nodata
            results = (
                {"properties": {"land_cover": int(v)}, "geometry": s}
                for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.transform))
            )
            geoms = list(results)
            gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)
            gdf.to_file(geojson_path, driver="GeoJSON")
