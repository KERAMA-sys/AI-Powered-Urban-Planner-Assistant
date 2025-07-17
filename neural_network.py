import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv("updated_dataset2.csv")
df["Speed_Deviation"] = df["Free_Flow_Speed"] - df["Current_Speed"]


# Splitting into features & target
X = df[["Latitude", "Longitude", "Day_Sin", "Month", "Season", "Temperature", "Wind_Speed", 
        "Weather_Code", "Confidence", "FRC", "hour_sin", "Minute", "Is_Peak_Hour", "Is_Weekend","Speed_Deviation" ]]


y = df["Speed_Ratio"]

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Define the MLP Regressor model
mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam',
                   learning_rate='adaptive', max_iter=1000, random_state=42)

# Train the model
mlp.fit(X_train_scaled, y_train)

# Predictions
y_pred = mlp.predict(X_test_scaled)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Print metrics
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")
print(f"MAPE: {mape}%")

# Save the trained model
with open("mlp_model.pkl", "wb") as f:
    pickle.dump(mlp, f)

print("Model saved as 'mlp_model.pkl' and scaler saved as 'scaler.pkl'")
