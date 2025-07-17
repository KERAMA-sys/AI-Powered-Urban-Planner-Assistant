import os
import geopandas as gpd
from shapely.geometry import mapping
from py3dep import get_dem
import numpy as np
import rioxarray
from tqdm import tqdm
import hashlib
from concurrent.futures import ProcessPoolExecutor

# Directories
input_dir = "geojson_cities_enriched"
output_dir = "geojson_with_elevation2"
dem_cache_dir = "dem_cache"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(dem_cache_dir, exist_ok=True)

# City-FIPS dictionary
city_fips2 = {
     "Philadelphia": "42101", "Pittsburgh": "42003","Chicago": "17031", "Minneapolis": "27053", "Detroit": "26163",
    "Cleveland": "39035", "Atlanta": "13089", "Houston": "48201", "Miami": "12086", "Charlotte": "37119",
    "New_Orleans": "22071", "Los_Angeles": "06037", "San_Francisco": "06075", "Seattle": "53033", "Denver": "08031",
    "Phoenix": "04013", "Las_Vegas": "32003", "Salt_Lake_City": "49035", "Dallas": "48113", "Austin": "48453",
    "San_Antonio": "48029", "Kansas_City": "29095", "Oklahoma_City": "40109", "Portland": "41051", "Tampa": "12103",
    "Baltimore": "24510", "St._Louis": "29510", "Indianapolis": "18097", "Raleigh": "37183", "Columbus": "39049",
    "Orlando": "12095", "Albuquerque": "35001", "Milwaukee": "55079", "Sacramento": "06067", "Richmond": "51760",
    "Cincinnati": "39061", "Nashville": "47037", "Buffalo": "36029", "San_Diego": "06073", "Jacksonville": "12031",
    "Memphis": "47157", "Louisville": "21111", "Fresno": "06019", "El_Paso": "48141", "Tucson": "04019",
    "Mesa": "04013", "Colorado_Springs": "08041", "Virginia_Beach": "51810", "Omaha": "31055", "Oakland": "06001",
    "Tulsa": "40143", "Arlington": "48439", "Bakersfield": "06029", "Wichita": "20173", "Aurora": "08005",
    "Honolulu": "15003", "Anchorage": "02020", "Boise": "16001"
}

def get_or_load_dem(bounds, resolution=30, crs="EPSG:4326"):
    """Fetch or load DEM from cache."""
    bbox_hash = hashlib.md5(str(bounds).encode()).hexdigest()
    cache_path = os.path.join(dem_cache_dir, f"{bbox_hash}.tif")

    if os.path.exists(cache_path):
        try:
            return rioxarray.open_rasterio(cache_path).squeeze()
        except Exception as e:
            print(f"⚠️ Failed to load cached DEM: {e}")

    try:
        dem = get_dem(bounds, resolution=resolution, crs=crs)
        dem.rio.to_raster(cache_path)
        return dem
    except Exception as e:
        print(f"❌ DEM fetch failed: {e}")
        return None

def enrich_with_elevation(gdf):
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    dem = get_or_load_dem((minx, miny, maxx, maxy))

    if dem is None:
        return None

    dem = dem.rio.write_crs("EPSG:4326", inplace=True)
    dem_array = dem.values
    gradient_x, gradient_y = np.gradient(dem_array)
    slope_array = np.degrees(np.arctan(np.sqrt(gradient_x**2 + gradient_y**2)))

    def zonal_stats(geom):
        try:
            clipped_dem = dem.rio.clip([mapping(geom)], crs="EPSG:4326", drop=True, all_touched=True)
            arr = clipped_dem.values
            if arr.size == 0 or np.isnan(arr).all():
                return np.nan, np.nan, np.nan, np.nan

            mean_elev = float(np.nanmean(arr))
            max_elev = float(np.nanmax(arr))
            min_elev = float(np.nanmin(arr))

            slope_da = dem.copy(data=slope_array)
            slope_da.rio.write_crs("EPSG:4326", inplace=True)
            clipped_slope = slope_da.rio.clip([mapping(geom)], crs="EPSG:4326", drop=True, all_touched=True)
            slope_vals = clipped_slope.values
            mean_slope = float(np.nanmean(slope_vals)) if slope_vals.size > 0 else np.nan

            return mean_elev, max_elev, min_elev, mean_slope

        except Exception as e:
            return np.nan, np.nan, np.nan, np.nan

    stats = gdf.geometry.apply(zonal_stats)
    gdf["mean_elevation"] = stats.apply(lambda x: x[0])
    gdf["max_elevation"] = stats.apply(lambda x: x[1])
    gdf["min_elevation"] = stats.apply(lambda x: x[2])
    gdf["mean_slope"] = stats.apply(lambda x: x[3])
    return gdf

def process_city(city):
    input_file = os.path.join(input_dir, f"{city}_USA_NLCD.geojson")
    output_file = os.path.join(output_dir, f"{city}_USA_NLCD_with_elevation.geojson")

    if not os.path.exists(input_file):
        return ("missing", city)

    try:
        gdf = gpd.read_file(input_file)
        if gdf.empty:
            return ("error", city, "GeoDataFrame is empty")

        enriched_gdf = enrich_with_elevation(gdf)
        if enriched_gdf is None:
            return ("error", city, "DEM fetch failed")

        enriched_gdf.to_file(output_file, driver="GeoJSON")
        return ("success", city)
    except Exception as e:
        return ("error", city, str(e))

# Main parallel run
if __name__ == "__main__":
    print("🚀 Starting elevation enrichment with multiprocessing and caching...\n")

    results = []
    with ProcessPoolExecutor(max_workers=9) as executor:
        for res in tqdm(executor.map(process_city, city_fips2.keys()), total=len(city_fips2)):
            results.append(res)

    # Summary
    missing = [r[1] for r in results if r[0] == "missing"]
    errors = [(r[1], r[2]) for r in results if r[0] == "error"]

    print("\n📊 Summary Report:")
    if missing:
        print("⚠️ Missing files:")
        for city in missing:
            print(f"  - {city}")
    if errors:
        print("🚨 Processing errors:")
        for city, err in errors:
            print(f"  - {city}: {err}")
    if not missing and not errors:
        print("✅ All cities processed successfully.")
