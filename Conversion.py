import os
import rasterio
import rasterio.mask
from rasterio.features import shapes
import geopandas as gpd
from tqdm import tqdm
import numpy as np
import osmnx as ox

# --------------------------
# Input and Output Directories
# --------------------------
NLCD_RASTER = "Annual_NLCD_LndCov_2023_CU_C1V0.tif"

# List of target U.S. cities
cities = [
         "Philadelphia, USA", "Pittsburgh, USA", "Chicago, USA", 
    "Minneapolis, USA", "Detroit, USA", "Cleveland, USA", "Atlanta, USA", "Houston, USA", 
    "Miami, USA", "Charlotte, USA", "New Orleans, USA", "Los Angeles, USA", "San Francisco, USA", 
    "Seattle, USA", "Denver, USA", "Phoenix, USA", "Las Vegas, USA", "Salt Lake City, USA", 
    "Dallas, USA", "Austin, USA", "San Antonio, USA", "Kansas City, USA", "Oklahoma City, USA", 
    "Portland, USA", "Tampa, USA", "Baltimore, USA", "St. Louis, USA", "Indianapolis, USA", 
    "Raleigh, USA", "Columbus, USA", "Orlando, USA", "Albuquerque, USA", "Milwaukee, USA", 
    "Sacramento, USA", "Richmond, USA", "Cincinnati, USA", "Nashville, USA", "Buffalo, USA", 
    "San Diego, USA", "Jacksonville, USA", "Memphis, USA", "Louisville, USA", "Fresno, USA", 
    "El Paso, USA", "Tucson, USA", "Mesa, USA", "Colorado Springs, USA", "Virginia Beach, USA", 
    "Omaha, USA", "Oakland, USA", "Tulsa, USA", "Arlington, USA", "Bakersfield, USA", 
    "Wichita, USA", "Aurora, USA", "Honolulu, USA", "Anchorage, USA", "Boise, USA"
]

# Output directories for clipped TIFFs and GeoJSONs
clipped_output_dir = "clipped_cities_nlcd"
geojson_output_dir = "geojson_cities_nlcd"
os.makedirs(clipped_output_dir, exist_ok=True)
os.makedirs(geojson_output_dir, exist_ok=True)

# --------------------------
# Open the NLCD raster once
# --------------------------
with rasterio.open(NLCD_RASTER) as src:
    for city_name in tqdm(cities):
        try:
            # Fetch city boundary using OSMnx
            print(f"\nFetching boundary for {city_name}...")
            city_boundary = ox.geocode_to_gdf(city_name)
            city_boundary = city_boundary.to_crs(src.crs)
            
            # Clip the NLCD raster with the city boundary
            print(f"Clipping raster for {city_name}...")
            out_image, out_transform = rasterio.mask.mask(src, city_boundary.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Save the clipped raster as a temporary TIF file
            tif_filename = city_name.replace(", ", "_").replace(" ", "_") + "_NLCD.tif"
            tif_output_path = os.path.join(clipped_output_dir, tif_filename)
            with rasterio.open(tif_output_path, "w", **out_meta) as dest:
                dest.write(out_image)
            print(f"✅ Saved clipped TIFF: {tif_output_path}")
            
            # --------------------------
            # Polygonize the Clipped Raster to GeoJSON
            # --------------------------
            with rasterio.open(tif_output_path) as clip_src:
                image = clip_src.read(1)
                mask_data = image != clip_src.nodata
                # Generate polygons from raster pixels while preserving the land cover value
                results = (
                    {"properties": {"land_cover": int(val)}, "geometry": geom}
                    for geom, val in shapes(image, mask=mask_data, transform=clip_src.transform)
                )
                geoms = list(results)
                if len(geoms) == 0:
                    print(f"⚠️ No features extracted for {city_name}.")
                    continue
                gdf = gpd.GeoDataFrame.from_features(geoms, crs=clip_src.crs)
                # Optionally clean geometries with a zero-width buffer (fixes minor issues)
                gdf["geometry"] = gdf.geometry.buffer(0)
                
                # Save to GeoJSON
                geojson_filename = city_name.replace(", ", "_").replace(" ", "_") + "_NLCD.geojson"
                geojson_output_path = os.path.join(geojson_output_dir, geojson_filename)
                gdf.to_file(geojson_output_path, driver="GeoJSON")
            print(f"✅ Saved GeoJSON: {geojson_output_path}")
            
        except Exception as e:
            print(f"❌ Failed for {city_name}: {e}")
