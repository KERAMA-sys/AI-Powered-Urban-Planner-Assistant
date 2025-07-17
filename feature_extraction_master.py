import os
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from shapely.ops import nearest_points
from scipy.spatial import cKDTree
from tqdm import tqdm
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from shapely.geometry import shape, Point
from scipy.spatial import KDTree


# -------------------------
# 📁 Directories
# -------------------------
input_dir = "geojson_cities_nlcd"
output_dir = "geojson_cities_enriched"
water_dir = "tiger_areawater_2023"
roads_dir = "tiger_roads_2023"
os.makedirs(output_dir, exist_ok=True)


# -------------------------
# 🌍 Load WorldPop Data
# -------------------------
print("📦 Loading WorldPop data...")
worldpop_df = pd.read_csv("worldcitiespop.csv", low_memory=False)
worldpop_df = worldpop_df.dropna(subset=["Latitude", "Longitude", "Population"])
worldpop_df["Population"] = pd.to_numeric(worldpop_df["Population"], errors="coerce")
worldpop_valid = worldpop_df.dropna(subset=["Population"]).copy()
worldpop_valid["coords"] = list(zip(worldpop_valid["Latitude"], worldpop_valid["Longitude"]))
city_tree = cKDTree(np.array(worldpop_valid["coords"].tolist()))
print("✅ WorldPop data loaded.\n")

# -------------------------
# 🧠 Helper Functions
# -------------------------
def get_nearest_distance_index(centroid, features):
    """
    Compute the distance from the given centroid to the nearest geometry in 'features'
    by iterating over each feature and taking the minimum distance.
    """
    # Check for empty features or invalid centroid
    if features.empty or centroid is None or not hasattr(centroid, "geom_type") or centroid.is_empty:
        return None
    try:
        # Compute distance for each geometry in features; ensure each geometry is valid
        distances = features.geometry.apply(
            lambda geom: centroid.distance(geom) if geom is not None and hasattr(geom, "geom_type") and not geom.is_empty else np.nan
        )
        # Drop any nan values and return the minimum distance, or None if all are nan
        valid_distances = distances.dropna()
        if valid_distances.empty:
            return None
        else:
            return valid_distances.min()
    except Exception as e:
        warnings.warn(f"Error computing distances individually: {e}")
        return None

def get_nearest_kdtree_distance(poly_centroid, kd_tree, coords):
    if poly_centroid is None or not poly_centroid.is_valid:
        return None
    dist, idx = kd_tree.query([poly_centroid.x, poly_centroid.y])
    nearest_point = Point(coords[idx])
    return poly_centroid.distance(nearest_point)

def get_population_from_centroid(lat, lon):
    _, idx = city_tree.query([lat, lon], k=1)
    return worldpop_valid.iloc[idx]["Population"]

# -------------------------
# 📍 FIPS Mapping
# -------------------------
city_fips = {
     "Los_Angeles": "06037", "Louisville": "21111", "Memphis": "47157", "Mesa": "04013",
    "Miami": "12086", "Milwaukee": "55079", "Minneapolis": "27053", "Nashville": "47037", "New_Orleans": "22071",
    "Oakland": "06001", "Oklahoma_City": "40109", "Omaha": "31055", "Orlando": "12095", "Philadelphia": "42101",
    "Phoenix": "04013", "Pittsburgh": "42003", "Portland": "41051", "Raleigh": "37183", "Richmond": "51760",
    "Sacramento": "06067", "Salt_Lake_City": "49035", "San_Antonio": "48029", "San_Diego": "06073",
    "San_Francisco": "06075", "Seattle": "53033", "St._Louis": "29510", "Tampa": "12103", "Tulsa": "40143",
    "Tucson": "04019", "Virginia_Beach": "51810", "Wichita": "20173"
}

# -------------------------
# 🚦 Sequential Processing
# -------------------------
print("🚀 Starting enrichment...\n")
for filename in tqdm(sorted(os.listdir(input_dir))):
    if not filename.endswith(".geojson"):
        continue

    city_name = filename.replace("_USA_NLCD.geojson", "")
    fips5 = city_fips.get(city_name, None)
    if not fips5:
        print(f"❌ Skipping: No FIPS for {city_name}")
        continue

    try:
        print(f"\n🌆 Processing {city_name}...")

        # Load city GeoJSON
        gdf = gpd.read_file(os.path.join(input_dir, filename)).to_crs(epsg=5070)
        gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
        gdf = gdf[gdf.geometry.area > 10]
        if gdf.empty:
            print(f"⚠️ Empty geometry for {filename}")
            continue

        # Area, perimeter, compactness
        gdf["area"] = gdf.geometry.area
        gdf["perimeter"] = gdf.geometry.length
        gdf["compactness"] = (gdf["perimeter"] ** 2) / (4 * np.pi * gdf["area"])

        # Centroid features
        gdf["centroid_geom"] = gdf.geometry.centroid
        gdf = gdf[gdf["centroid_geom"].apply(lambda x: x.is_valid and not x.is_empty)]
        centroids_wgs84 = gdf["centroid_geom"].to_crs(epsg=4326)
        gdf["centroid_lat"] = centroids_wgs84.y
        gdf["centroid_lon"] = centroids_wgs84.x

        # Create bounding box
        buffer = 10000
        minx, miny, maxx, maxy = gdf.total_bounds
        bbox_geom = box(minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=gdf.crs)

        # Load Water shapefile
        water_path = os.path.join(water_dir, f"tl_2023_{fips5}_areawater", f"tl_2023_{fips5}_areawater.shp")
        if os.path.exists(water_path):
            water_gdf = gpd.read_file(water_path).to_crs(epsg=5070)
            print(f"💧 Loaded water shapefile for {city_name}")
        else:
            water_gdf = gpd.GeoDataFrame(geometry=[])
            print(f"⚠️ Water shapefile not found for {city_name}")

        # Load Roads shapefile
        roads_path = os.path.join(roads_dir, f"tl_2023_{fips5}_roads", f"tl_2023_{fips5}_roads.shp")
        if os.path.exists(roads_path):
            roads_gdf = gpd.read_file(roads_path).to_crs(epsg=5070)
            print(f"🛣️ Loaded roads shapefile for {city_name}")
        else:
            roads_gdf = gpd.GeoDataFrame(geometry=[])
            print(f"⚠️ Roads shapefile not found for {city_name}")

        # Distance to water
        print("📏 Calculating distance to water...")

         # Quick check for water_gdf
        if water_gdf.empty:
            print("⚠️ No water geometries found, skipping water distance...")
            gdf["dist_to_water_m"] = None
        else:
            print(f"🌊 Loaded {len(water_gdf)} water geometries.")
            print("🧪 Validating water geometries...")
            water_gdf = water_gdf[water_gdf.geometry.is_valid & ~water_gdf.geometry.is_empty]
            if water_gdf.empty:
                print("⚠️ All water geometries invalid or empty.")
                gdf["dist_to_water_m"] = None
            else:
                distances = []
                for i, row in tqdm(gdf.iterrows(), total=len(gdf), desc="→ Water Distance"):
                    try:
                        centroid = row["centroid_geom"]
                        if centroid is None or centroid.is_empty or not centroid.is_valid:
                            distances.append(None)
                        else:
                            dist = get_nearest_distance_index(centroid, water_gdf)
                            distances.append(dist)
                    except Exception as e:
                        print(f"❌ Error for index {i}: {e}")
                        distances.append(None)
                gdf["dist_to_water_m"] = distances

        print("✅ Distance to water calculated.")

        # Distance to road
        print("🚧 Calculating distance to roads (with spatial index)...")

        if roads_gdf.empty:
            print("⚠️ No road geometries found, skipping road distance...")
            gdf["dist_to_road_m"] = None
        else:
            roads_gdf = roads_gdf[roads_gdf.geometry.is_valid & ~roads_gdf.geometry.is_empty]
            road_centroids = roads_gdf.geometry.centroid
            coords = np.array([[pt.x, pt.y] for pt in road_centroids])
            kd_tree = KDTree(coords)
            distances_to_road = []
            for i, row in tqdm(gdf.iterrows(), total=len(gdf), desc="→ Road Distance (KDTree)"):
                try:
                    centroid = row["centroid_geom"]
                    if centroid is None or centroid.is_empty or not centroid.is_valid:
                        distances_to_road.append(None)
                    else:
                        dist = get_nearest_kdtree_distance(centroid, kd_tree, coords)
                        distances_to_road.append(dist)
                except Exception as e:
                    print(f"❌ Error at index {i}: {e}")
                    distances_to_road.append(None)

            # Add result to GeoDataFrame
            gdf["dist_to_road_m"] = distances_to_road
            print("✅ Distance to roads calculated.")

        # Population density
        print("👥 Estimating population density...")
        def safe_population_lookup(row):
            try:
                return get_population_from_centroid(row["centroid_lat"], row["centroid_lon"])
            except Exception as e:
                warnings.warn(f"Population lookup error: {e}")
                return None

        gdf["population_density"] = gdf.apply(safe_population_lookup, axis=1)
        print("✅ Population density assigned.")


        # Final cleanup
        gdf["city"] = city_name
        gdf.drop(columns=["centroid_geom"], inplace=True)
        gdf.set_geometry("geometry", inplace=True)
        gdf = gdf.to_crs(epsg=4326)
        gdf.to_file(os.path.join(output_dir, filename), driver="GeoJSON")

        print(f"✅ Finished {filename}")

    except Exception as e:
        print(f"❌ Error processing {filename}: {str(e)}")

print("\n🎉 All cities processed.")
