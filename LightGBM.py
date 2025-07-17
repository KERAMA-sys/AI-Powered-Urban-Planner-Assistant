import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
import lightgbm as lgb
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load your dataset
df = pd.read_csv("Consolidated_Land_Use_Data.csv")

# ✅ Encode labels
le = LabelEncoder()
df['zoning_encoded'] = le.fit_transform(df['zoning_label'])

# ✅ Features and labels
X = df.drop(columns=['zoning_label', 'zoning_encoded','land_cover'])
y = df['zoning_encoded']

# ✅ Compute class weights
classes = np.unique(y)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weights = dict(zip(classes, weights))

# ✅ Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ✅ Convert to LightGBM Dataset with weights
train_weights = np.array([class_weights[label] for label in y_train])
train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
valid_data = lgb.Dataset(X_test, label=y_test)

# ✅ Parameters
params = {
    'objective': 'multiclass',
    'num_class': len(classes),
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'max_depth': 10,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'boosting_type': 'gbdt'
}

# ✅ Train with callbacks for early stopping
print("⏳ Training LightGBM model...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
)

# ✅ Predict
print("🔍 Evaluating...")
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_classes = np.argmax(y_pred, axis=1)

# ✅ Metrics
acc = accuracy_score(y_test, y_pred_classes)
macro_f1 = f1_score(y_test, y_pred_classes, average='macro')

print(f"✅ Final Accuracy: {acc:.4f}")
print(f"✅ Final Macro F1 Score: {macro_f1:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

print(set(df.columns) & set(X.columns))


corr = df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# ✅ Save model & label encoder
joblib.dump(model, 'lgbm_zoning_model.pkl')
joblib.dump(le, 'zoning_label_encoder.pkl')
