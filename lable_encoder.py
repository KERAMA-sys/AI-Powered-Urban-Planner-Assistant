import pandas as pd
import pickle
from sklearn.impute import KNNImputer

# Load dataset
df = pd.read_csv("air_pollution_model/global_air_pollution.csv")

# Define a **global** AQI category mapping
aqi_category_mapping = {
    "Good": 0,
    "Moderate": 1,
    "Unhealthy for Sensitive Groups": 2,
    "Unhealthy": 3,
    "Very Unhealthy": 4,
    "Hazardous": 5
}

# Columns that need encoding
category_columns = [
    "AQI Category",
    "CO AQI Category",
    "Ozone AQI Category",
    "NO2 AQI Category",
    "PM2.5 AQI Category"
]

# **Step 1: Check Missing Values Before Encoding**
print("\n🔹 Missing Values Before Encoding:")
print(df.isnull().sum())  # Checks for both NaN and null values

# Encode categorical columns
for col in category_columns:
    if col in df.columns:
        df[col] = df[col].map(aqi_category_mapping)

# **Step 2: Handle remaining NaN values (if encoding created any)**
df.dropna(subset=category_columns, inplace=True)  # Removes rows where category columns are NaN

# **Step 3: Check Missing Values After Encoding**
print("\n🔹 Missing Values After Encoding:")
print(df.isnull().sum())

# Save the encoding mapping
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(aqi_category_mapping, f)

# Save updated dataset
df.to_csv("encoded_dataset.csv", index=False)

print("✅ Encoding applied successfully! Null & NaN values")

# Load your dataset
file_path = "air_pollution_model/encoded_dataset.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Select only Latitude and Longitude columns for imputation
imputer = KNNImputer(n_neighbors=5)  # Adjust k as needed
df[['Latitude', 'Longitude']] = imputer.fit_transform(df[['Latitude', 'Longitude']])

# Save the updated dataset
output_file = "updated_dataset.csv"
df.to_csv(output_file, index=False)

print(f"Updated dataset saved as {output_file}")




# # Decoder for the label encoders


# import pickle

# # Load encoding mapping
# with open("aqi_category_mapping.pkl", "rb") as f:
#     aqi_category_mapping = pickle.load(f)

# # Reverse mapping for decoding
# reverse_mapping = {v: k for k, v in aqi_category_mapping.items()}

# # Example prediction
# predicted_category = 3  # Replace with your model's output
# decoded_category = reverse_mapping[predicted_category]

# print("Decoded AQI Category:", decoded_category)