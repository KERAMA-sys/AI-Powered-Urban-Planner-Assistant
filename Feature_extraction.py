import numpy as np
import pandas as pd
import requests
import datetime
from timezonefinder import TimezoneFinder
from shapely.geometry import Polygon, shape
from shapely.ops import transform
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import pyproj
import geopandas as gpd
import py3dep
from functools import partial
import overpy
import os   
import sys

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..','delivarables','backend'))

# Construct paths
SCALARS_DIR = os.path.join(base_dir, 'scalars')
MODELS_DIR = os.path.join(base_dir, 'models')

# Load models and scalers
traffic_scaler = joblib.load(os.path.join(SCALARS_DIR, "traffic_scalar.pkl"))
pollution_scaler = joblib.load(os.path.join(SCALARS_DIR, "air_pollution_scalar.pkl"))
pollution_model = joblib.load(os.path.join(MODELS_DIR, "pollution_model.pkl"))
with open(os.path.join(SCALARS_DIR, "pollution_labels.pkl"), "rb") as f:
    pollution_label_mapping = pickle.load(f)
reverse_pollution_mapping = {v: k for k, v in pollution_label_mapping.items()}

overpass_api = overpy.Overpass()

# === Geo Helper ===
def get_centroid(polygon_coords):
    """
    Calculate the centroid of a polygon.
    Expects polygon_coords in (latitude, longitude) format.
    """
    try:
        poly = Polygon([(lon, lat) for lat, lon in polygon_coords])
        if poly.is_empty:
            raise ValueError("Cannot compute centroid of empty polygon")
            
        centroid = poly.centroid
        lat, lon = centroid.y, centroid.x  # Shapely returns (x,y) which is (lon,lat)
        
        # Validate latitude
        if lat < -90 or lat > 90:
            print(f"⚠️ Invalid centroid latitude: {lat}. Adjusting to valid range.")
            lat = max(min(lat, 90), -90)
            
        return lat, lon
    except Exception as e:
        print(f"Error calculating centroid: {e}")
        # Return a default value or raise an exception
        raise

def get_timezone(lat, lon):
    tf = TimezoneFinder()
    return tf.timezone_at(lng=lon, lat=lat) or "UTC"


# === Feature Extraction Modules ===
def extract_traffic_features(lat, lon):
    """
    Extract traffic features for a given latitude and longitude.
    """
    # Load the current time and calculate time-based features
    now = datetime.datetime.utcnow()
    local_tz = get_timezone(lat, lon)
    local_time = now.astimezone(datetime.timezone.utc).astimezone()
    hour = local_time.hour
    weekday = local_time.weekday()
    hour_sin = round(np.sin(2 * np.pi * hour / 24), 6)
    day_sin = round(np.sin(2 * np.pi * weekday / 7), 6)
    is_peak = 1 if hour in [8, 9, 17, 18] else 0
    is_weekend = 1 if weekday in [5, 6] else 0
    month = local_time.month
    minute = local_time.minute

    # Fetch weather data from Open-Meteo API
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        weather_response = requests.get(weather_url)
        weather_response.raise_for_status()
        weather_data = weather_response.json().get("current_weather", {})
        temperature = weather_data.get("temperature", 0)
        wind_speed = weather_data.get("windspeed", 0)
        weather_code = weather_data.get("weathercode", 0)
    except requests.RequestException as e:
        print(f"❌ Weather API request failed: {e}")
        temperature = 0
        wind_speed = 0
        weather_code = 0

    frc_mapping = {
    "FRC0": 0, "FRC1": 1, "FRC2": 2, "FRC3": 3, "FRC4": 4,
    "FRC5": 5, "FRC6": 6, "FRC7": 7  # Default is 7 (Minor Local Roads)
    }
    # Fetch traffic data from TomTom API
    TOMTOM_API_KEY = "TaKVpj4FLGV2CwAc6pKfpEbJFzqnpOWr"
    TOMTOM_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/18/json"
    tomtom_url = f"{TOMTOM_URL}?key={TOMTOM_API_KEY}&point={lat},{lon}"
    try:
        traffic_response = requests.get(tomtom_url)
        traffic_response.raise_for_status()
        traffic_data = traffic_response.json().get("flowSegmentData", {})
        confidence = traffic_data.get("confidence", 0.0)
        current_speed = traffic_data.get("currentSpeed", 1)
        free_flow_speed = max(traffic_data.get("freeFlowSpeed", 1), 1)
        frc= traffic_data.get("frc", "FRC7")  # Default FRC value if not found
        actual_speed_ratio = current_speed / free_flow_speed
        speed_deviation = free_flow_speed - current_speed
        frc =frc_mapping.get(frc, 7)  # Map FRC to integer value, default to 7 if not found
    except requests.RequestException as e:
        print(f"❌ Traffic API request failed: {e}")
        confidence = 0.0
        actual_speed_ratio = 1.0
        speed_deviation = 0.0
        frc = 7  # Default FRC value

    # Determine the season based on the month
    season = get_season(month)

    # Combine all features in the correct order
    features = pd.DataFrame([{
        "Latitude": lat,
        "Longitude": lon,
        "Day_Sin": day_sin,
        "Month": month,
        "Season": season,
        "Temperature": temperature,
        "Wind_Speed": wind_speed,
        "Weather_Code": weather_code,
        "Confidence": confidence,
        "FRC": frc,
        "hour_sin": hour_sin,
        "Minute": minute,
        "Is_Peak_Hour": is_peak,
        "Is_Weekend": is_weekend,
        "Speed_Deviation": speed_deviation
    }])

    # Scale the features
    scaled = traffic_scaler.transform(features)

    return {
        "features": scaled.tolist(),
        "raw": features.to_dict(orient="records")[0]
    }


def get_season(month):
    """Returns season based on the month."""
    if month in [12, 1, 2]: return 1  # Winter
    elif month in [3, 4, 5]: return 2  # Spring
    elif month in [6, 7, 8]: return 3  # Summer
    elif month in [9, 10, 11]: return 4  # Autumn
    return 1

def extract_pollution_features(lat, lon):
    timezone = get_timezone(lat, lon)
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone", "us_aqi"],
        "timezone": timezone
    }
    try:
        response = requests.get(url, params=params)
        if  response.status_code != 200:
            raise Exception(f"Pollution API Error: {response.status_code}, {response.text}")
        data = response.json()["current"]

        # Convert pollutant values using correct conversion formulas
        co_ppm = round(data["carbon_monoxide"] / 1145, 6)  # Convert CO from µg/m³ to ppm
        ozone_ppb = round(data["ozone"] / 1.96, 6)         # Convert Ozone from µg/m³ to ppb
        no2 = round(data["nitrogen_dioxide"], 6)           # NO2 in µg/m³
        pm2_5 = round(data["pm2_5"], 6)                    # PM2.5 in µg/m³

        # Create a DataFrame to ensure feature order matches the model's training data
        df = pd.DataFrame([[lat, lon, co_ppm, ozone_ppb, no2, pm2_5]],
                          columns=["Latitude", "Longitude", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"])
        scaled = pollution_scaler.transform(df)
        prediction = pollution_model.predict(scaled)[0]

        return {
            "features": scaled.tolist(),
            "raw": {
                "Latitude": lat,
                "Longitude": lon,
                "CO AQI Value": co_ppm,
                "Ozone AQI Value": ozone_ppb,
                "NO2 AQI Value": no2,
                "PM2.5 AQI Value": pm2_5
            },
            "prediction_encoded": int(prediction),
            "prediction_decoded": reverse_pollution_mapping.get(prediction, "Unknown")
        }
    except requests.RequestException as e:
        print(f"❌ Pollution API request failed: {e}")
        return {
            "features": [],
            "raw": {},
            "prediction_encoded": None,
            "prediction_decoded": "Error"
        }

# Supporting Functions

def calculate_area_perimeter_compactness(geom):
    """
    Calculate the area, perimeter, and compactness of a polygon.
    """
    try:

        if not geom.is_valid:
            print("⚠️ Invalid geometry detected. Attempting to fix with buffer(0)...")
            geom = geom.buffer(0)

        # Use GeoPandas for accurate area and perimeter calculations
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        gdf = gdf.to_crs("EPSG:3857")  # Reproject to a metric CRS for accurate calculations
        gdf["area"] = gdf.geometry.area  # Area in square meters
        gdf["perimeter"] = gdf.geometry.length  # Perimeter in meters
        gdf["compactness"] = (gdf["perimeter"] ** 2) / (4 * np.pi * gdf["area"])  # Compactness formula

        # Extract calculated values
        area = float(gdf.geometry.area.iloc[0])
        perim = float(gdf.geometry.length.iloc[0])

        if area > 0:
            comp = (perim ** 2) / (4 * np.pi * area)
        else:
            comp = 0
            
        return area, perim, comp
    

    except Exception as e:
        print(f"Error calculating area, perimeter, and compactness: {e}")
        return 0, 0, 0  # Return default values instead of None

def get_distances(lat, lon):
    """
    Calculate the distances to the nearest water and road features using Overpass API.
    """
    # First validate latitude is within valid range
    if lat < -90 or lat > 90:
        print(f"⚠️ Invalid latitude value: {lat}. Adjusting to valid range.")
        lat = max(min(lat, 90), -90)  # Clamp between -90 and 90
        
    def nearest(tag):
        try:
            # Query Overpass API for the specified tag (e.g., "waterway", "highway")
            query = f"""
            [out:json];
            (
              way["{tag}"](around:1000,{lat},{lon});
              node["{tag}"](around:1000,{lat},{lon});
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
            return min(distances) if distances else None
        except Exception as e:
            print(f"Error fetching distance for tag '{tag}': {e}")
            return None

    # Calculate distances to water and roads
    dist_to_water = nearest("waterway") or nearest("water")  # Try both "waterway" and "water"
    dist_to_road = nearest("highway")  # Use "highway" for roads

    return dist_to_water, dist_to_road

# def get_distances(lat, lon):
    """
    Calculate the distances to the nearest water and road features using Overpass API.
    """
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
    """
    Fetch elevation statistics (mean, max, min, slope) for the given polygon.
    """
    try:
        # Get the bounding box of the polygon
        minx, miny, maxx, maxy = geom.bounds

        # Fetch DEM data for the bounding box
        dem = py3dep.get_dem((minx, miny, maxx, maxy), resolution=30)

        # Extract elevation values
        values = dem.data.flatten() if hasattr(dem, 'data') else dem.values.flatten()
        values = values[~np.isnan(values)]  # Remove NaN values

        # If no valid elevation data is found, return None
        if values.size == 0:
            return None, None, None, None

        # Calculate slope using the gradient of elevation values
        slope = np.gradient(values)

        # Return mean, max, min elevation, and mean slope
        return float(values.mean()), float(values.max()), float(values.min()), float(np.mean(slope))
    except Exception as e:
        print(f"Error fetching elevation stats: {e}")
        return None, None, None, None


def map_land_cover_from_tags(tags):
    """
    Map land cover tags from Overpass API to a zoning label.
    """
    tag_value = tags.get('landuse') or tags.get('natural')
    if tag_value:
        if tag_value in ['forest', 'wood']:
            return 'Green Space'
        if tag_value in ['residential']:
            return 'Residential'
        if tag_value in ['commercial', 'retail', 'industrial']:
            return 'Commercial'
        if tag_value in ['meadow', 'farmland', 'orchard']:
            return 'Agricultural'
        if tag_value in ['wetland']:
            return 'Wetlands'
        if tag_value in ['water', 'reservoir']:
            return 'Open Water'
        if tag_value in ['scrub', 'grassland']:
            return 'Agricultural'
        if tag_value in ['barren', 'desert']:
            return 'Barren'
    return 'Unknown'


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
def map_land_cover_to_number(land_cover_label):
    return zoning_to_nlcd.get(land_cover_label, 0)

# Main Function


def extract_land_use_features(polygon_coords,lat, lon):
    """
    Extract land-use features for a given polygon.
    """
    
    # Create polygon and validate geometry
    # poly = Polygon([(lon, lat) for lat, lon in polygon_coords])

    # Create polygon and validate geometry
    polygon_coords_transformed = [(coord[1], coord[0]) for coord in polygon_coords]
    poly = Polygon(polygon_coords)
    # Ensure the polygon is closed
    if polygon_coords[0] != polygon_coords[-1]:
        raise ValueError("Polygon coordinates must form a closed loop (first and last points must be the same).")

    # Calculate area, perimeter, and compactness
    area, perim, comp = calculate_area_perimeter_compactness(poly)
    area = area if area is not None else 0
    perim = perim if perim is not None else 0
    comp = comp if comp is not None else 0

    # Get centroid
    latc, lonc = lat,lon

    # Query Overpass for landuse/natural at the centroid point
    poly_str = ' '.join(f"{pt[1]} {pt[0]}" for pt in polygon_coords)
    query = f"""
    [out:json];
    (
      way(around:100.0,{latc},{lonc})["landuse"];
      way(around:100.0,{latc},{lonc})["natural"];
      way(poly:"{poly_str}")["building"];
      node(poly:"{poly_str}")["building"];
    );
    out body;
    """
    result = overpass_api.query(query)
    land_cover = 'Unknown'
    land_cover = 'Unknown'
    for w in result.ways:
        if 'landuse' in w.tags or 'natural' in w.tags:
            land_cover = map_land_cover_from_tags(w.tags)
            break
    land_cover = map_land_cover_to_number(land_cover)

    # 7) Building count and population density proxy
    building_ways = [w for w in result.ways if 'building' in w.tags]
    building_nodes = result.nodes  # only building nodes queried
    building_count = len(building_ways) + len(building_nodes)
    popden = building_count / (area / 1e6) if area > 0 else 0

    # if result.ways:
    #     way = result.ways[0]  # Take the first hit
    #     land_cover = map_land_cover_from_tags(way.tags)

    # land_cover = map_land_cover_to_number(land_cover)

    # # 6) Building count and population density proxy
    # counts = result.counts
    # building_count = counts.get('ways', 0) + counts.get('nodes', 0)
    # popden = building_count / (area / 1e6) if area > 0 else 0

    # Get distances, population density, and elevation stats
    dwater, droad = get_distances(latc, lonc)
    # popden = 64090.0  # Example population density
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
        'land_cover': land_cover,
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

    return features

# === Main Unified Extractor ===
def extract_all_features_from_polygon(polygon_coords):
    # lat, lon = get_centroid(polygon_coords)
    polygon_coords_transformed = [(coord[1], coord[0]) for coord in polygon_coords]
    print(f"Transformed Polygon Coordinates: {polygon_coords_transformed}")
    lat,lon = get_centroid(polygon_coords_transformed)
    print(f"Centroid: {lat}, {lon}")
    return {
        "centroid": {"lat": lat, "lon": lon},
        "Traffic": extract_traffic_features(lat, lon),
        "Pollution": extract_pollution_features(lat, lon),
        "LandUse": extract_land_use_features(polygon_coords,lat,lon)
    }

# === Example for Testing ===
# if __name__ == "__main__":
    # polygon = [
    #     [41.879, -87.632], [41.879, -87.627],
    #     [41.874, -87.627], [41.874, -87.632],
    #     [41.879, -87.632]
    # ]
    # polygon = [[-122.442829, 37.792034], 
    #                     [-122.431439, 37.792034],
    #                     [-122.431439, 37.801034],
    #                     [-122.442829, 37.801034], 
    #                     [-122.442829, 37.792034]]

    # For San Francisco area (correct coordinates)
    # polygon = [
    #     [-122.442829, 37.792034],  # [lon, lat]
    #     [-122.431439, 37.792034],
    #     [-122.431439, 37.801034], 
    #     [-122.442829, 37.801034],
    #     [-122.442829, 37.792034]  # Same as first point to close the polygon
    # ]

    # result = extract_all_features_from_polygon(polygon)
    # print(result)