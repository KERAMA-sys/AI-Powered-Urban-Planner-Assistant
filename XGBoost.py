from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import cross_val_score
import shap

# Load Data
df = pd.read_csv("air_pollution_model/scaled_dataset.csv")


# Features & Labels
X = df[["Latitude", "Longitude","CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"]]
y = df["AQI Category"]

# Convert data to float32 for optimized GPU memory usage
X = X.astype(np.float32)

# Handle Imbalanced Data using ADASYN
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Hyperparameter Grid
param_dist = {
    "n_estimators": [200, 300, 400],
    "max_depth": [3, 4],
    "learning_rate": [0.25, 0.05],
    "min_child_weight": [10, 11, 12],
    "gamma": [0.6, 0.7, 0.8],
    "subsample": [0.5, 0.6, 0.7],
    "colsample_bytree": [0.5, 0.6, 0.7],
    "reg_lambda": [50,60,70]
}

# XGBoost Model with GPU Support
xgb_model = XGBClassifier(eval_metric="mlogloss", tree_method="hist")


# Hyperparameter Tuning with Stratified K-Fold
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    xgb_model, param_dist, n_iter=20, cv=cv_strategy, verbose=1, n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)

# Train Final Model with Best Parameters
final_model = XGBClassifier(
    **random_search.best_params_, eval_metric="mlogloss", tree_method="hist")
final_model.fit(X_train, y_train)

# Predictions
y_pred = final_model.predict(X_test)

# Evaluation
train_pred = final_model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print("Training Accuracy:", train_acc)
accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Balanced Accuracy:", balanced_acc)
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)

# Cross-Validation Scores
cv_scores = cross_val_score(final_model, X_resampled, y_resampled, cv=cv_strategy, scoring="accuracy")

print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())

# Save Model
joblib.dump(final_model, "XGBoost_model.pkl")
print("Final AQI Model Saved Successfully!")

# Real-Time Prediction Function
def predict_final_aqi(aqi_values):
    model = joblib.load("XGBoost_model.pkl")
    
    return model.predict([aqi_values])[0]

# Example Prediction
scaler = joblib.load("air_pollution_model/scaler.pkl")
real_time_data = np.array[28.7041,77.1025,4, 50, 30, 60]  # Example CO, Ozone, NO2, PM2.5 AQI values
columns = ["Latitude", "Longitude", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"]
data_df=pd.DataFrame(real_time_data)
data_df_scale=scaler.transform(data_df)
real_time_data_scaled_df = pd.DataFrame(data_df_scale, columns=columns)
predicted_aqi = predict_final_aqi(real_time_data_scaled_df)
print("Predicted Final AQI Category:", predicted_aqi)

# Feature Correlation Matrix
corr_matrix = X_train.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# Feature Importance
importances = final_model.feature_importances_
feature_names = X_train.columns

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color="skyblue")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance (XGBoost)")
plt.show()

# SHAP Feature Importance
explainer = shap.TreeExplainer(final_model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
