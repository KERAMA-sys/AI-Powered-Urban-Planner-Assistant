import requests
import joblib
import pandas as pd
import pickle
import numpy as np
from timezonefinder import TimezoneFinder

# Load trained AQI model
model1 = joblib.load("air_pollution_model/RF_model.pkl")
model2=joblib.load("air_pollution_model/XGBoost_model.pkl")
# Load scaler
scaler = joblib.load("air_pollution_model/scaler.pkl")

# Load AQI category mapping
with open("air_pollution_model/Lable_encoders.pkl", "rb") as f:
    aqi_category_mapping = pickle.load(f)

# Reverse mapping for decoding AQI category
reverse_mapping = {v: k for k, v in aqi_category_mapping.items()}

# Function to get timezone based on latitude & longitude
def get_timezone(lat, lon):
    tf = TimezoneFinder()
    return tf.timezone_at(lng=lon, lat=lat)

# Coordinates (Modify as needed)
latitude = 41.8781
longitude = -87.6298

# Get the timezone dynamically
timezone = get_timezone(latitude, longitude)
if not timezone:
    timezone = "UTC"  # Default fallback

# API endpoint
url = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Request current air pollution data
params = {
    "latitude": latitude,
    "longitude": longitude,
    "current": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone", "us_aqi"],
    "timezone": timezone
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()["current"]

    # Convert pollutant values using correct conversion formulas
    co_ppm = data["carbon_monoxide"] / 1145  # Convert CO from µg/m³ to ppm
    ozone_ppb = data["ozone"] / 1.96          # Convert Ozone from µg/m³ to ppb (updated factor)

    # Feature order must match training data
    columns = ["Latitude", "Longitude", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"]
    
    # Create DataFrame for real-time data
    pollutant_values = np.array([[latitude, longitude, co_ppm, ozone_ppb, data["nitrogen_dioxide"], data["pm2_5"]]])
    real_time_data_df = pd.DataFrame(pollutant_values, columns=columns)

    # ✅ Debug: Print raw input data before scaling
    print("\n🔹 Raw Input Data Before Scaling:")
    print(real_time_data_df)

    # Check Min/Max values used in training
    print("\n🔹 Min/Max Values Used for Scaling:")
    print("Min Values:", scaler.data_min_)
    print("Max Values:", scaler.data_max_)

    # Scale pollutant values using the loaded scaler
    pollutant_values_scaled = scaler.transform(real_time_data_df)
    real_time_data_scaled_df = pd.DataFrame(pollutant_values_scaled, columns=columns)

    # ✅ Debug: Print scaled input data
    print("\n🔹 Scaled Data Passed to Model:")
    print(real_time_data_scaled_df)

    # Predict AQI Category (Encoded)
    predicted_category = model1.predict(real_time_data_scaled_df)[0]
    predicted_category2=model2.predict(real_time_data_scaled_df)[0]
    # Decode AQI Category
    decoded_category = reverse_mapping.get(predicted_category, "Unknown")
    decoded_category2=reverse_mapping.get(predicted_category2,"Unknown")
    # ✅ Debug: Print label mappings
    print("\n🔹 AQI Category Mapping (Encoded to Decoded):")
    print(reverse_mapping)

    # ✅ Debug: Feature importance (if available)
    if hasattr(model1, "feature_importances_"):
        print("\n🔹 Feature Importance:")
        for feature, importance in zip(columns, model1.feature_importances_):
            print(f"{feature}: {importance:.4f}")
    else:
        print("\n🔹 Feature Importance:2")
        for feature, importance in zip(columns, model2.feature_importances_):
            print(f"{feature}: {importance:.4f}")
    # Print Results
    print("\n===== INPUT DATA FROM API (Converted) =====")
    print(f"Time Zone: {timezone}")
    print(f"CO AQI Value (ppm): {co_ppm:.4f}")
    print(f"Ozone AQI Value (ppb): {ozone_ppb:.2f}")
    print(f"NO2 AQI Value: {data['nitrogen_dioxide']}")
    print(f"PM2.5 AQI Value: {data['pm2_5']}")
    print("Expected Scaled PM2.5:", (29.3 - 0) / (500 - 0))

    print("\n===== PREDICTION =====")
    print(f"Predicted AQI Category (Encoded): {predicted_category}")
    print(f"Predicted AQI Category (Decoded): {decoded_category}")
    print(f"Predicted AQI Category (Encoded): {predicted_category2}")
    print(f"Predicted AQI Category (Decoded): {decoded_category2}")

else:
    print(f"API Error: {response.status_code}, Response: {response.text}")
