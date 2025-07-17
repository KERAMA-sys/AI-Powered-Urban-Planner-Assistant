import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import shap
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data
df = pd.read_csv("updated_dataset2.csv")
df.dropna(inplace=True)  # Drop rows with missing values

# Feature Engineering
df["Speed_Deviation"] = df["Free_Flow_Speed"] - df["Current_Speed"]

# Features & Target
X = df[["Latitude", "Longitude", "Day_Sin", "Month", "Season", "Temperature", "Wind_Speed", 
        "Weather_Code", "Confidence", "FRC", "hour_sin", "Minute", "Is_Peak_Hour", "Is_Weekend","Speed_Deviation" ]]
y = df["Speed_Ratio"]

# Convert data to float32 for optimized memory usage
X = X.astype(np.float32)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 800, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 6, 14),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.02, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "gamma": trial.suggest_float("gamma", 0.0, 0.3),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
        "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 15.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 2.0, 10.0)
    }
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model without early stopping
    model = xgb.XGBRegressor(**params, eval_metric="rmse", tree_method="hist")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

# Run Optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Best Parameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Train Final Model with K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
final_model = xgb.XGBRegressor(**best_params, eval_metric="rmse", tree_method="hist")

best_rmse = float("inf")
stagnant_rounds = 0
patience = 10  # Custom early stopping patience

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = final_model.predict(X_test)
    current_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    if current_rmse < best_rmse:
        best_rmse = current_rmse
        stagnant_rounds = 0  # Reset counter if improvement found
    else:
        stagnant_rounds += 1  # Increase counter if no improvement
        if stagnant_rounds >= patience:
            print("Early stopping triggered.")
            break

# Save Model
joblib.dump(final_model, "XGBoost_Traffic_Model.pkl")
print("Final Traffic Model Saved Successfully!")

# Predictions
y_pred = final_model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance Metrics:")
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R² Score:", round(r2, 4))

# Cross-Validation Score
scores = cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")
cv_rmse = np.sqrt(-scores.mean())
print("Cross-Validation RMSE:", cv_rmse)

# Feature Importance using SHAP (Optimized for Performance)
explainer = shap.Explainer(final_model)
shap_values = explainer(X_test[:500])  # Limiting to 500 samples for efficiency
shap.summary_plot(shap_values, X_test[:500])

# Feature Importance using XGBoost
xgb.plot_importance(final_model, importance_type="gain")
plt.show()

# Feature Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(X_train.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# Target Distribution Check
plt.figure(figsize=(8, 5))
sns.histplot(df["Speed_Ratio"], kde=True, bins=30)
plt.title("Target Variable Distribution (Speed Ratio)")
plt.show()
