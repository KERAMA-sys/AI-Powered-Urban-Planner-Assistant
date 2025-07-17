import os
import numpy as np
import pandas as pd
import joblib
import pyproj
from shapely.geometry import Polygon
from shapely.ops import transform
from functools import partial
import overpy
import py3dep
import geopandas as gpd

# === Paths & Constants ===
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(base_dir, 'models')
scalar = os.path.join(base_dir, 'scalars')
OUTPUT_CSV = os.path.join(base_dir, 'outputs', 'Land_Use_Features.csv')

# === Load Land Use Model & Encoder ===
lgbm_model = joblib.load(os.path.join(MODELS_DIR, 'lgbm_zoning_model.pkl'))
le = joblib.load(os.path.join(scalar, 'zoning_label_encoder.pkl'))

overpass_api = overpy.Overpass()

# === NLCD to Zoning Mapping ===
nlcd_to_zoning = {
    11: "Open Water",
    21: "Residential",
    22: "Residential",
    23: "Commercial",
    24: "Commercial",
    31: "Barren",
    41: "Green Space",
    42: "Green Space",
    43: "Green Space",
    52: "Agricultural",
    71: "Agricultural",
    81: "Agricultural",
    82: "Agricultural",
    90: "Wetlands",
    95: "Wetlands"
}

zoning_to_nlcd = {v: k for k, v in nlcd_to_zoning.items()}

# === Helper Functions ===
def calculate_area_perimeter_compactness(geom):
    try:
        # Calculate area, perimeter, and compactness using GeoPandas
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        gdf = gdf.to_crs("EPSG:3857")  # Reproject to a metric CRS for accurate calculations
        gdf["area"] = gdf.geometry.area  # Area in square meters
        gdf["perimeter"] = gdf.geometry.length  # Perimeter in meters
        gdf["compactness"] = (gdf["perimeter"] ** 2) / (4 * np.pi * gdf["area"])  # Compactness formula

        # Extract calculated values
        area = gdf["area"].iloc[0]
        perim = gdf["perimeter"].iloc[0]
        comp = gdf["compactness"].iloc[0]
        return area, perim, comp
    except Exception:
        return None, None, None

def get_distances(lat, lon):
    def nearest(tag):
        try:
            # Query Overpass API for the specified tag (e.g., "waterway", "highway")
            query = f"""
            [out:json];
            (
              way["{tag}"](around:5000,{lat},{lon});
              node["{tag}"](around:5000,{lat},{lon});
            );
            out center;
            """
            result = overpass_api.query(query)

            # Collect all points (nodes and way centers)
            points = []
            for node in result.nodes:
                points.append((float(node.lat), float(node.lon)))
            for way in result.ways:
                if hasattr(way, 'center_lat') and hasattr(way, 'center_lon'):
                    points.append((float(way.center_lat), float(way.center_lon)))

            # If no points are found, return None
            if not points:
                return None

            # Calculate the minimum distance to any point
            distances = [np.hypot(lat - p[0], lon - p[1]) * 111000 for p in points]  # Convert degrees to meters
            return min(distances)
        except Exception as e:
            print(f"Error fetching distance for tag '{tag}': {e}")
            return None

    # Calculate distances to water and roads
    dist_to_water = nearest("waterway") or nearest("water")  # Try both "waterway" and "water"
    dist_to_road = nearest("highway")  # Use "highway" for roads

    return dist_to_water, dist_to_road

def get_elevation_stats(geom):
    minx, miny, maxx, maxy = geom.bounds
    dem = py3dep.get_dem((minx, miny, maxx, maxy), resolution=30)
    try:
        values = dem.data.flatten()
    except AttributeError:
        values = dem.values.flatten()
    values = values[~np.isnan(values)]
    if values.size == 0:
        return None, None, None, None
    slope = np.gradient(values)
    return float(values.mean()), float(values.max()), float(values.min()), float(np.mean(slope))

def map_land_cover_from_tags(tags):
    tag_value = tags.get('landuse') or tags.get('natural')
    if tag_value:
        if tag_value in ['forest', 'wood']: return 'Green Space'
        if tag_value in ['residential']: return 'Residential'
        if tag_value in ['commercial', 'retail', 'industrial']: return 'Commercial'
        if tag_value in ['meadow', 'farmland', 'orchard']: return 'Agricultural'
        if tag_value in ['wetland']: return 'Wetlands'
        if tag_value in ['water', 'reservoir']: return 'Open Water'
        if tag_value in ['scrub', 'grassland']: return 'Agricultural'
        if tag_value in ['barren', 'desert']: return 'Barren'
    return 'Unknown'

def map_land_cover_to_number(land_cover_label):
    return zoning_to_nlcd.get(land_cover_label, 0)  # Default to 0 if not found

# === Land Use Feature Extraction ===
def extract_land_use_features(polygon_coords):
    # Create polygon and validate geometry
    poly = Polygon([(lon, lat) for lat, lon in polygon_coords])
    if not poly.is_valid:
        print("⚠️ Invalid polygon geometry detected. Attempting to fix...")
        poly = poly.buffer(0)  # Fix invalid geometries
    if not poly.is_valid:
        raise ValueError("Polygon geometry could not be fixed. Please check the input coordinates.")

    # Ensure the polygon is closed
    if polygon_coords[0] != polygon_coords[-1]:
        raise ValueError("Polygon coordinates must form a closed loop (first and last points must be the same).")

    # Calculate area, perimeter, and compactness
    area, perim, comp = calculate_area_perimeter_compactness(poly)
    area = area if area is not None else 100342
    perim = perim if perim is not None else 45342
    comp = comp if comp is not None else 34534

    # Get centroid
    centroid = poly.centroid
    latc, lonc = centroid.y, centroid.x

    # Query Overpass for landuse/natural at the centroid point
    query = f"""
    [out:json];
    (
      way(around:100.0,{latc},{lonc})["landuse"];
      way(around:100.0,{latc},{lonc})["natural"];
    );
    out body;
    """
    result = overpass_api.query(query)
    land_cover = 'Unknown'

    if result.ways:
        way = result.ways[0]  # Take the first hit
        land_cover = map_land_cover_from_tags(way.tags)

    land_cover = map_land_cover_to_number(land_cover)

    # Get distances, population density, and elevation stats
    dwater, droad = get_distances(latc, lonc)
    popden = 64090.0  # Example population density
    me, ma, mi, ms = get_elevation_stats(poly)

    # Set standard values for missing data
    dwater = float(dwater) if dwater is not None else 200
    droad = float(droad) if droad is not None else 500
    me = float(me) if me is not None else 0
    ma = float(ma) if ma is not None else 0
    mi = float(mi) if mi is not None else 0
    ms = float(ms) if ms is not None else 0

    # Debugging: Print intermediate values
    print(f"Area: {area}, Perimeter: {perim}, Compactness: {comp}")
    print(f"Distances - Water: {dwater}, Road: {droad}")
    print(f"Elevation - Mean: {me}, Max: {ma}, Min: {mi}, Slope: {ms}")

    # Create feature dictionary
    features = {
        'land_cover': int(land_cover),
        'area': float(area),
        'perimeter': float(perim),
        'compactness': float(comp),
        'centroid_lat': float(latc),
        'centroid_lon': float(lonc),
        'dist_to_water_m': float(dwater),
        'dist_to_road_m': float(droad),
        'population_density': float(popden),
        'mean_elevation': float(me),
        'max_elevation': float(ma),
        'min_elevation': float(mi),
        'mean_slope': float(ms)
    }

    # Create DataFrame and make predictions
    df = pd.DataFrame([features])
    if not df.empty:
        preds = lgbm_model.predict(df)
        print(f"Shape of predictions: {preds.shape}")  # Debugging
        if len(preds.shape) > 1:  # Multi-class predictions
            predicted_class = np.argmax(preds, axis=1)  # Get the most likely class
            df['zoning_label'] = le.inverse_transform(predicted_class.astype(int))
        else:  # Single prediction
            if len(preds) != len(df):
                raise ValueError(f"Mismatch between predictions ({len(preds)}) and DataFrame rows ({len(df)}).")
            df['zoning_label'] = le.inverse_transform(preds.astype(int).ravel())
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
    
    return df

# === Test ===
if __name__ == '__main__':
    sample = [
        [35.21809528136627, -106.72779975054371],
        [35.219, -106.726],
        [35.217, -106.725],
        [35.216, -106.728],
        [35.21809528136627, -106.72779975054371]
    ]
    df = extract_land_use_features(sample)
    print("\n🌍 Extracted Features for Polygon Centroid:\n")
    print(df.to_string(index=False))

