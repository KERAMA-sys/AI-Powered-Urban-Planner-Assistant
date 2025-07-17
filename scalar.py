import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the dataset
data = pd.read_csv('air_pollution_model/encoded_dataset.csv')

# Define the columns to normalize
columns_to_normalize = [
     "Latitude","Longitude","CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"
]

# Initialize the MinMaxScaler (you can also use StandardScaler if preferred)
scaler = MinMaxScaler()

# Normalize the specified columns
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# Save the scaler as scaler.pkl
joblib.dump(scaler, 'scaler.pkl')

# Optionally, save the normalized dataset
data.to_csv('Final_dataset.csv', index=False)

print("Normalization complete and scaler.pkl saved.")
