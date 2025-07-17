import time
import requests
import pandas as pd
import os
import joblib
import pickle
import numpy as np
from datetime import datetime
import pytz
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
MODEL_PATH = "mlp_model.pkl"
SCALER_PATH = "scaler.pkl"
TOMTOM_API_KEY = "TaKVpj4FLGV2CwAc6pKfpEbJFzqnpOWr"
TOMTOM_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/18/json"

def load_model():
    """Load the trained MLPRegressor model and scaler."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

def fetch_real_time_data(lat, lon, timezone):
    """Fetch real-time traffic and weather data for given coordinates."""
    tomtom_url = f"{TOMTOM_URL}?key={TOMTOM_API_KEY}&point={lat},{lon}"
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    
    try:
        traffic_response = requests.get(tomtom_url)
        weather_response = requests.get(weather_url)
        traffic_response.raise_for_status()
        weather_response.raise_for_status()
        
        traffic_data = traffic_response.json().get("flowSegmentData", {})
        weather_data = weather_response.json().get("current_weather", {})
        
        if not traffic_data or not weather_data:
            return None
        
        city_tz = pytz.timezone(timezone)
        local_time = datetime.now(city_tz)
        
        return {
            "Latitude": lat,
            "Longitude": lon,
            "Day_Sin": round(np.sin(2 * np.pi * local_time.weekday() / 7), 6),
            "Month": local_time.month,
            "Season": get_season(local_time.month),
            "Temperature": weather_data.get("temperature", 0),
            "Wind_Speed": weather_data.get("windspeed", 0),
            "Weather_Code": weather_data.get("weathercode", 0),
            "Confidence": traffic_data.get("confidence", 0.0),
            "FRC": get_frc_from_api(lat, lon),  # Fetch FRC dynamically
            "hour_sin": round(np.sin(2 * np.pi * local_time.hour / 24), 6),
            "Minute": local_time.minute,
            "Is_Peak_Hour": 1 if local_time.hour in [8, 9, 17, 18] else 0,
            "Is_Weekend": 1 if local_time.weekday() in [5, 6] else 0,
            "Actual_Speed_Ratio": traffic_data.get("currentSpeed", 1) / max(traffic_data.get("freeFlowSpeed", 1), 1),
            "Speed_Deviation":traffic_data.get("freeFlowSpeed", 1) - traffic_data.get("currentSpeed", 1)

        }
    except requests.RequestException as e:
        print(f"❌ API request failed: {e}")
        return None

def get_season(month):
    """Returns season based on the month."""
    if month in [12, 1, 2]: return 1  # Winter
    elif month in [3, 4, 5]: return 2  # Spring
    elif month in [6, 7, 8]: return 3  # Summer
    elif month in [9, 10, 11]: return 4  # Autumn
    return 1

def get_frc_from_api(lat, lon):
    """Placeholder function for fetching Functional Road Class (FRC)."""
    return 3  # Default FRC value

def preprocess_data(data, scaler):
    """Ensure real-time data matches trained feature format before prediction."""
    feature_order = ["Latitude", "Longitude", "Day_Sin", "Month", "Season", "Temperature",
                     "Wind_Speed", "Weather_Code", "Confidence", "FRC", "hour_sin", "Minute", "Is_Peak_Hour", "Is_Weekend", "Speed_Deviation"]
    df = pd.DataFrame([data])
    df = df[feature_order]  # Ensure correct feature order
    return scaler.transform(df)  # Normalize features

def test_model():
    """Fetch real-time data, process it, make a prediction, and compare it with actual traffic data."""
    model, scaler = load_model()
    lat, lon, timezone = 28.6139, 77.2090, "Asia/Kolkata"  # Example: Delhi, India
    
    data = fetch_real_time_data(lat, lon, timezone)
    if data:
        actual_speed_ratio = data.pop("Actual_Speed_Ratio")  # Remove actual value from feature set
        processed_data = preprocess_data(data, scaler)
        prediction = model.predict(processed_data)[0]
        
        print("✅ Real-time data fetched:", data)
        print(f"🚀 Predicted Speed Ratio: {prediction:.3f}")
        print(f"📊 Actual Speed Ratio: {actual_speed_ratio:.3f}")
        print(f"📉 Prediction Error: {abs(prediction - actual_speed_ratio):.3f}")
        
        if abs(prediction - actual_speed_ratio) > 0.15:
            print("⚠️ Warning: Model may need recalibration for better accuracy.")
        else:
            print("✅ Model prediction is close to real-time data.")
    else:
        print("❌ Failed to fetch real-time data.")

# Run the test
test_model()
