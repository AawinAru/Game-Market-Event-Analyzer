import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

print("BASE_DIR:", BASE_DIR)
print("DATA_PROCESSED exists:", DATA_PROCESSED.exists())

# ✅ Load ML dataset
ml_data = pd.read_csv(DATA_PROCESSED / "ml_dataset.csv", sep=";")
print(f"✅ Loaded ml_dataset.csv: {ml_data.shape}")

# Separate features and target
X = ml_data.drop(columns=["impact_label", "impact_label_num"])
y = ml_data["impact_label_num"]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ✅ Train Gradient Boosting (best model)
print("\n🤖 Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb.fit(X_train, y_train)

# Evaluate
y_pred = gb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ F1-score: {f1:.4f}")

# ✅ Save the model
MODEL_PATH = DATA_PROCESSED / "best_model.pkl"
joblib.dump(gb, MODEL_PATH)
print(f"\n✅ Model saved to: {MODEL_PATH}")

# ✅ Also save feature names for later use
feature_names = X.columns.tolist()
joblib.dump(feature_names, DATA_PROCESSED / "feature_names.pkl")
print(f"✅ Feature names saved")

print("\n🎯 Now you can use train_models.py to make predictions!")