import time
import schedule
import requests
import pandas as pd
import os
from datetime import datetime
import subprocess
import pytz

# API Key for TomTom Traffic API (Replace with your key)
TOMTOM_API_KEYS = ["TaKVpj4FLGV2CwAc6pKfpEbJFzqnpOWr","R2XnGEhky4rfWGyjR5P1qc8D6PDOGJxK"]
GEONAMES_USERNAME = "anish001" # geonames username 
current_api_index=0


def get_active_tomtom_key():
    """Return the currently active TomTom API key."""
    return TOMTOM_API_KEYS[current_api_index]

def switch_tomtom_key():
    """Switch to the next TomTom API key when rate limit is hit."""
    global current_api_index
    current_api_index = (current_api_index + 1) % len(TOMTOM_API_KEYS)
    print(f"🔄 Switching to TomTom API Key: {get_active_tomtom_key()}")


# Cities for Data Collection
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

def determine_season(month, latitude):
    """Determine the season based on the month and hemisphere."""
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

def fetch_weather_data(lat, lon):
    """Fetch weather data from Open-Meteo API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "current_weather" in data:
            weather = data["current_weather"]
            return weather["temperature"], weather["windspeed"], weather["weathercode"]
        else:
            return None, None, None
    except requests.RequestException as e:
        print(f"❌ Weather API request failed: {e}")
        return None, None, None


# Define file paths dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
cities_file_path = os.path.join(script_dir, "cities.txt")
csv_file_path = os.path.join(script_dir, "traffic_data.csv")
log_file_path = os.path.join(script_dir, "traffic_errors.log")

cities = load_cities_from_file(cities_file_path)
print(f"✅ Loaded {len(cities)} major cities from file.")

# Store previous values for cities
previous_speeds = {}
previous_free_flows = {}

# FRC Mapping
frc_mapping = {
    "FRC0": 0, "FRC1": 1, "FRC2": 2, "FRC3": 3, "FRC4": 4,
    "FRC5": 5, "FRC6": 6, "FRC7": 7  # Default is 7 (Minor Local Roads)
}

# Default speeds by road type (km/h)
frc_default_speeds = {
    0: 110, 1: 90, 2: 70, 3: 60, 4: 50, 5: 40, 6: 30, 7: 25
}

def fetch_data_with_retry(url, retries=3, delay=5):
    """Fetch API data with retry mechanism."""
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if "flowSegmentData" in data:
                return data["flowSegmentData"]
            elif response.status_code == 429:
                print("⚠️ Rate limit reached. Switching API key...")
                switch_tomtom_key()
            else:
                print(f"⚠️ Unexpected API response: {data}")
                return None
        except requests.RequestException as e:
            print(f"❌ Request failed: {e}")
        
        print(f"⚠️ Retrying {attempt+1}/{retries}...")
        time.sleep(delay)
    
    print(f"❌ Final attempt failed for {url}")
    return None

def fetch_traffic_data():
    global previous_speeds, previous_free_flows
    
    traffic_data = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"📊 Starting data collection at {timestamp}")

    for city, (lat, lon,timezone) in cities.items():
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/18/json?key={get_active_tomtom_key()}&point={lat},{lon}"
        data = fetch_data_with_retry(url)
        temperature, windspeed, weather_code = fetch_weather_data(lat, lon) # Fetch weather data

        if data and all(value is not None for value in (temperature, windspeed, weather_code)):
           
            print(f"✅ Data received for {city}")
            # Get local time, UTC time, day of week, month, and season
            city_tz = pytz.timezone(timezone)
            local_time = datetime.now(city_tz)
            utc_time = local_time.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
            day_of_week = local_time.strftime("%A")
            month = local_time.strftime("%B")
            season = determine_season(local_time.month, lat)

            # Extract values from API response
            current_speed = data.get("currentSpeed")
            free_flow_speed = data.get("freeFlowSpeed")
            current_travel_time = data.get("currentTravelTime")  # in seconds
            free_flow_travel_time = data.get("freeFlowTravelTime")  # in seconds
            confidence = data.get("confidence", 0.0)  # API confidence score
            frc_code = data.get("frc", "FRC7")
            frc_value = frc_mapping.get(frc_code, 7)

            # Handle missing current_speed
            if current_speed == 0.0:
                if free_flow_speed:
                    current_speed = previous_speeds.get(city, free_flow_speed * 0.85)
                else:
                    default_speed = frc_default_speeds.get(frc_value, 30)
                    current_speed = previous_speeds.get(city, default_speed * 0.85)
                print(f"⚠️ Missing 'currentSpeed' for {city}, estimated {current_speed:.1f} km/h")
            
            # Handle missing free_flow_speed
            if free_flow_speed == 0.0:
                if current_speed:
                    multiplier = 1.3 if frc_value < 3 else 1.2
                    free_flow_speed = previous_free_flows.get(city, current_speed * multiplier)
                else:
                    free_flow_speed = previous_free_flows.get(city, frc_default_speeds.get(frc_value, 30))
                print(f"⚠️ Missing 'freeFlowSpeed' for {city}, estimated {free_flow_speed:.1f} km/h")
            
            # Store values for next run
            previous_speeds[city] = current_speed
            previous_free_flows[city] = free_flow_speed
            
            
            
            # Speed ratio for ML models
            # Speed ratio for ML models
            speed_ratio = round(current_speed / free_flow_speed, 3) if free_flow_speed > 0 else 0

            
            # Save traffic data with essential metrics for ML
            traffic_data.append({
                "Timestamp": timestamp,
                "Latitude": lat,
                "Longitude": lon,
                "Day_of_Week": day_of_week,
                "Month": month,
                "Season": season,
                "Temperature": temperature,
                "Wind_Speed": windspeed,
                "Weather_Code": weather_code,
                "Current_Speed": round(current_speed, 1),
                "Free_Flow_Speed": round(free_flow_speed, 1),
                "Speed_Ratio": round(speed_ratio, 3),
                "Current_Travel_Time": current_travel_time,
                "Free_Flow_Travel_Time": free_flow_travel_time,
                "Confidence": round(confidence, 2),
                "FRC": frc_value
            })
        else:
            # Log failures
            print(f"❌ Failed to fetch data for {city}")
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"[{timestamp}] Failed to fetch traffic data for {city}\n")

    # Save to CSV
    if traffic_data:
        df = pd.DataFrame(traffic_data)
        print("\n📊 Sample data:")
        print(df[["Day_of_Week", "Month", "Season", "Temperature", "Wind_Speed", "Weather_Code", "Current_Speed", "Speed_Ratio", "Free_Flow_Speed", "FRC", "Confidence"]].head())
        df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)
        print(f"✅ Traffic data saved at {timestamp}. Collected {len(traffic_data)} records.")
    else:
        print("❌ No traffic data collected in this run.")


print("🔄 Traffic data collection service started...")

def notify_progress():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🔔 Reminder: Traffic data collection running. Last update at {timestamp}")
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}]  Data collection still active.\n")

fetch_traffic_data()
# Schedule Tasks
schedule.every(10).minutes.do(fetch_traffic_data)
schedule.every(30).minutes.do(notify_progress)

start_time = time.time()
while True:
    schedule.run_pending()
    time.sleep(10)
