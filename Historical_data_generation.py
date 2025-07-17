import requests
import pandas as pd
import random
from datetime import datetime, timedelta
import pytz
import time
import os

# -------------------------------
# Configuration & City Definitions
# -------------------------------
def load_cities_from_file(file_path):
    """Load major cities and their coordinates from a text file."""
    cities = {}
    try:
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split(",")
                if len(parts) == 4:
                    name, lat, lon, timezone = parts
                    cities[name] = (float(lat), float(lon), timezone)
    except FileNotFoundError:
        print("❌ Error: cities.txt file not found!")
    return cities

# Load cities from cities.txt and convert the dictionary to a list of tuples.
script_dir = os.path.dirname(os.path.abspath(__file__))
cities_file_path = os.path.join(script_dir, "cities.txt")
cities_dict = load_cities_from_file(cities_file_path)
cities = [(name, lat, lon, timezone) for name, (lat, lon, timezone) in cities_dict.items()]

# Target number of records to generate
TARGET_RECORDS = 100000

# Date range for synthetic data (example: year 2024)
START_DATE = datetime(2024, 1, 1, 0, 0, 0)
END_DATE   = datetime(2024, 12, 31, 23, 59, 59)
TOTAL_DAYS = (END_DATE - START_DATE).days

# -------------------------------
# Weather Cache to Avoid Duplicate API Calls
# -------------------------------
# Cache key: (lat, lon, date_str), value: tuple (time_objs, temperatures, windspeeds, weathercodes)
weather_cache = {}

# -------------------------------
# Helper Functions
# -------------------------------
def determine_season(month, latitude):
    """Determine the season based on month and hemisphere."""
    if latitude >= 0:  # Northern Hemisphere
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"
    else:  # Southern Hemisphere
        if month in [12, 1, 2]:
            return "Summer"
        elif month in [3, 4, 5]:
            return "Autumn"
        elif month in [6, 7, 8]:
            return "Winter"
        else:
            return "Spring"

def weather_speed_multiplier(weather_code):
    """
    Return a multiplier to adjust traffic speeds based on weather conditions.
    Multiplier values are chosen based on Open-Meteo documentation:
      - 0: Clear sky → 1.0
      - 1,2,3: Mainly clear/partly cloudy/overcast → 0.95
      - 45,48: Fog → 0.90
      - 51,53,55: Drizzle → 0.90
      - 56,57: Freezing drizzle → 0.85
      - 61,63,65: Rain → 0.80
      - 66,67: Freezing rain → 0.75
      - 71,73,75,77: Snow fall → 0.75
      - 80,81,82: Rain showers → 0.85
      - 95,96,99: Thunderstorms → 0.70
    """
    if weather_code == 0:
        return 1.0
    elif weather_code in [1, 2, 3]:
        return 0.95
    elif weather_code in [45, 48]:
        return 0.90
    elif weather_code in [51, 53, 55]:
        return 0.90
    elif weather_code in [56, 57]:
        return 0.85
    elif weather_code in [61, 63, 65]:
        return 0.80
    elif weather_code in [66, 67]:
        return 0.75
    elif weather_code in [71, 73, 75, 77]:
        return 0.75
    elif weather_code in [80, 81, 82]:
        return 0.85
    elif weather_code in [95, 96, 99]:
        return 0.70
    else:
        return 1.0

def fetch_historical_weather_data(lat, lon, timestamp_dt):
    """
    Fetch historical weather data from Open-Meteo archive API for the given timestamp.
    It queries for the date corresponding to the timestamp and then finds the hourly observation
    closest to the timestamp.
    Implements caching to avoid duplicate API calls.
    """
    date_str = timestamp_dt.strftime("%Y-%m-%d")
    cache_key = (lat, lon, date_str)
    
    if cache_key in weather_cache:
        # Use cached data
        time_objs, temperatures, windspeeds, weathercodes = weather_cache[cache_key]
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": ["temperature_2m", "windspeed_10m", "weathercode"]
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temperatures = hourly.get("temperature_2m", [])
            windspeeds = hourly.get("windspeed_10m", [])
            weathercodes = hourly.get("weathercode", [])
            if times and temperatures and windspeeds and weathercodes:
                # Convert API times (e.g., "2024-01-01T00:00") to datetime objects
                time_objs = [datetime.strptime(t, "%Y-%m-%dT%H:%M") for t in times]
                # Cache the result
                weather_cache[cache_key] = (time_objs, temperatures, windspeeds, weathercodes)
            else:
                # Fallback default values if API returns no data
                print(f"⚠️ No weather data found for lat {lat}, lon {lon} on {date_str}")
                return 15.0, 10.0, 0
        except Exception as e:
            print(f"❌ Historical weather API error for lat {lat}, lon {lon} on {date_str}: {e}")
            return 15.0, 10.0, 0
    
    # Find index of the closest hour to the timestamp_dt using cached time_objs
    closest_index = min(range(len(time_objs)), key=lambda i: abs(time_objs[i] - timestamp_dt))
    return temperatures[closest_index], windspeeds[closest_index], weathercodes[closest_index]

def simulate_traffic_params(frc_value, base_current, base_free_flow, weather_code):
    """
    Adjust base speeds based on weather using the multiplier.
    Also simulate current and free-flow travel times and confidence.
    """
    multiplier = weather_speed_multiplier(weather_code)
    adjusted_current_speed = base_current * multiplier
    adjusted_free_flow_speed = base_free_flow * multiplier

    # Ensure free-flow speed is always above current speed
    if adjusted_free_flow_speed <= adjusted_current_speed:
        adjusted_free_flow_speed = adjusted_current_speed * (1.0 + (0.2 if frc_value < 3 else 0.15))

    speed_ratio = round(adjusted_current_speed / adjusted_free_flow_speed, 3)

    # Simulate travel times (in seconds) based on speeds: lower speeds imply higher travel times.
    current_travel_time = random.randint(60, 600)
    free_flow_travel_time = max(current_travel_time - random.randint(5, 60), 1)

    # Confidence is kept high (0.9-1.0)
    confidence = round(random.uniform(0.9, 1.0), 2)

    return round(adjusted_current_speed, 1), round(adjusted_free_flow_speed, 1), speed_ratio, current_travel_time, free_flow_travel_time, confidence

# Default speeds by Functional Road Classification (FRC)
# (base_current_speed, base_free_flow_speed)
frc_default_speeds = {
    0: (110, 120),
    1: (90, 105),
    2: (70, 85),
    3: (60, 75),
    4: (50, 65),
    5: (40, 55),
    6: (30, 45),
    7: (25, 35)
}

# For simulation, randomly select an FRC value (0 to 7)
def random_frc():
    return random.randint(0, 7)

# -------------------------------
# Synthetic Data Generation
# -------------------------------
synthetic_records = []
record_count = 0
output_csv = "generated_traffic_data.csv"

while record_count < TARGET_RECORDS:
    # Randomly select a city from the list loaded from cities.txt
    
    city_name, lat, lon, timezone = random.choice(cities)
        
        # Generate a random timestamp within the date range
    random_day = random.randint(0, TOTAL_DAYS)
    random_seconds = random.randint(0, 86399)
    timestamp_dt = START_DATE + timedelta(days=random_day, seconds=random_seconds)
        
        # Adjust timestamp to the city's timezone and extract day, month, season
    city_tz = pytz.timezone(timezone)
    local_dt = timestamp_dt.astimezone(city_tz)
    day_of_week = local_dt.strftime("%A")
    month_str = local_dt.strftime("%B")
    season = determine_season(local_dt.month, lat)
        
        # Fetch historical weather data for this timestamp
    temperature, wind_speed, weather_code = fetch_historical_weather_data(lat, lon, timestamp_dt)
        
        # Select a random FRC value and get default speeds for that road class
    frc_value = random_frc()
    base_current_speed, base_free_flow_speed = frc_default_speeds.get(frc_value, (30, 40))
        
        # Introduce slight randomness in the base speeds
    base_current_speed *= random.uniform(0.9, 1.1)
    base_free_flow_speed *= random.uniform(0.9, 1.1)
        
        # Adjust speeds based on weather conditions and simulate other traffic parameters
    current_speed, free_flow_speed, speed_ratio, current_travel_time, free_flow_travel_time, confidence = \
        simulate_traffic_params(frc_value, base_current_speed, base_free_flow_speed, weather_code)
        
        # Append the record in the required format
    synthetic_records.append({
            "Timestamp": timestamp_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Latitude": lat,
            "Longitude": lon,
            "Day_of_Week": day_of_week,
            "Month": month_str,
            "Season": season,
            "Temperature": temperature,
            "Wind_Speed": wind_speed,
            "Weather_Code": weather_code,
            "Current_Speed": current_speed,
            "Free_Flow_Speed": free_flow_speed,
            "Speed_Ratio": speed_ratio,
            "Current_Travel_Time": current_travel_time,
            "Free_Flow_Travel_Time": free_flow_travel_time,
            "Confidence": confidence,
            "FRC": frc_value
    })
        
    record_count += 1
    if record_count % 1000 == 0:
        print(f"{record_count} records generated...")

    # Convert to DataFrame and save to CSV
df = pd.DataFrame(synthetic_records)
df.to_csv(output_csv, index=False, mode='a', header=not os.path.exists(output_csv))


print(f"✅ Generated {len(df)} synthetic records and saved to '{output_csv}'")
