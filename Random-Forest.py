from matplotlib import pyplot as plt
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn
import os
# -------------------------
# Load Data and Preprocess
# -------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
# Construct the full path to the dataset 
dataset_path = os.path.join(base_dir, "air_pollution_model", "Final_dataset.csv")

# Load the dataset
df = pd.read_csv(dataset_path)

# Verify Column Names
print("Columns in DataFrame:", df.columns.tolist())

# Define numerical columns (including coordinates if you want them as features)
num_cols = ["Latitude", "Longitude", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"]

# Separate Features & Target
X = df[num_cols]
y = df["AQI Category"]

# Handle Imbalanced Classes using SMOTE if the target has few unique classes
if y.nunique() < 10:
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

# ------------------------
# Train-Test Split & CV Setup
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() < 10 else None
)

# Optionally, define a StratifiedKFold for further cross-validation in calibration, etc.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ------------------------------------
# Hyperparameter Tuning using GridSearchCV
# ------------------------------------
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [6, 8,9],
    "min_samples_split": [10, 20, 30],
    "min_samples_leaf": [10, 15, 20],
}

rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
grid_search = GridSearchCV(
    rf_model, param_grid, cv=5, verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train Final RandomForest Model with best parameters
final_rf_model = RandomForestClassifier(**best_params, random_state=42, class_weight="balanced")
final_rf_model.fit(X_train, y_train)

# -----------------------------------------------------------
# Calibrate the Model using CalibratedClassifierCV (isotonic regression)
# -----------------------------------------------------------
calibrated_rf = CalibratedClassifierCV(final_rf_model, method='isotonic', cv=skf)
calibrated_rf.fit(X_train, y_train)

# Save the calibrated model
joblib.dump(calibrated_rf, "air_pollution_model/final_rf_model_calibrated.pkl")
print("Final Calibrated AQI Model Saved Successfully!")

# ---------------
# Evaluation
# ---------------
y_pred = calibrated_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance Analysis from the uncalibrated model (calibration doesn't change importances)
plt.figure(figsize=(8, 5))
importances = final_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [X.columns[i] for i in indices], rotation=90)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

# -----------------------
# Predict AQI for new real-time data
# -----------------------
# Load the scaler (make sure this scaler was fit on the pollutant columns during training)
scaler = joblib.load("air_pollution_model/scaler.pkl")

# Define new real-time sample input:
# Features order: [Latitude, Longitude, CO AQI Value, Ozone AQI Value, NO2 AQI Value, PM2.5 AQI Value]
real_time_data = np.array([[28.7041, 77.1025, 1, 36, 0, 51]])

# Transform only the pollutant columns if those were scaled (assuming Latitude & Longitude were scaled if included)
# For this example, we assume the scaler was fit on all num_cols.
real_time_data_scaled = scaler.transform(real_time_data)

# Predict using the calibrated model
predicted_aqi_value = calibrated_rf.predict(real_time_data_scaled)[0]
print(f"Predicted AQI Category: {predicted_aqi_value}")
