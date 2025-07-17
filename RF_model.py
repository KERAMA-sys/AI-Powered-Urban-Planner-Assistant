import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('updated_dataset3.csv')

# Define features and target
X = df[['Latitude', 'Longitude', 'Day_of_Week', 'Month', 'Season', 'Temperature', 'Wind_Speed', 'Weather_Code', 'Current_Travel_Time', 'Free_Flow_Travel_Time', 'Confidence', 'FRC', 'Hour', 'Minute']]
y = df['Current_Speed']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [6, 8, 9],
    "min_samples_split": [ 20, 30,50],
    "min_samples_leaf": [10, 15, 20],
}

# Initialize RandomForestRegressor and perform grid search
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train final model with best parameters
final_rf_model = RandomForestRegressor(**best_params, random_state=42)
final_rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(final_rf_model, 'traffic_rf_model.pkl')

# Model Evaluation
y_pred = final_rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-Squared Score (R2):", r2)

# Feature Importance
feature_importance = final_rf_model.feature_importances_
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance of RandomForestRegressor')
plt.show()

print("Model training complete and saved as 'traffic_rf_model.pkl'")
