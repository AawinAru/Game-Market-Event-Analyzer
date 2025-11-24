"""Classify event impact magnitude using ML."""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODEL_PATH = DATA_PROCESSED / "best_model.pkl"

# ✅ Check if model exists, if not train it
if MODEL_PATH.exists():
    print(f"✅ Loading existing model from {MODEL_PATH}")
    best_model = joblib.load(MODEL_PATH)
else:
    print("⚠️ Model not found. Training now...")
    
    # Load and train
    ml_data = pd.read_csv(DATA_PROCESSED / "ml_dataset.csv", sep=";")
    X = ml_data.drop(columns=["impact_label", "impact_label_num"])
    y = ml_data["impact_label_num"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    best_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    best_model.fit(X_train, y_train)
    
    # Save for future use
    joblib.dump(best_model, MODEL_PATH)
    print(f"✅ Model trained and saved to {MODEL_PATH}")


def classify_event_impact(
    market_return: float,
    ar_event: float,
    car_0_1: float,
    car_m1_p1: float,
    car_0_3: float,
    car_0_5: float,
    car_m5_p5: float,
    is_rockstar: int = 0,
    sentiment_negative: int = 0,
    sentiment_neutral: int = 0,
    publisher_ea: int = 0,
    event_type_release: int = 0,
    franchise_gta: int = 0
) -> dict:
    """Predict event market impact using trained Gradient Boosting model."""
    
    features = [[
        market_return, ar_event, car_0_1, car_m1_p1, car_0_3, car_0_5, car_m5_p5,
        is_rockstar, sentiment_negative, sentiment_neutral, publisher_ea,
        event_type_release, franchise_gta
    ]]
    
    prediction = best_model.predict(features)[0]
    probabilities = best_model.predict_proba(features)[0]
    
    impact_labels = ["LOW", "MEDIUM", "HIGH"]
    
    return {
        "impact": impact_labels[int(prediction)],
        "confidence": round(float(probabilities[int(prediction)]) * 100, 2),
        "probabilities": {
            "low": round(float(probabilities[0]) * 100, 2),
            "medium": round(float(probabilities[1]) * 100, 2),
            "high": round(float(probabilities[2]) * 100, 2)
        }
    }


if __name__ == "__main__":
    result = classify_event_impact(
        market_return=0.015,
        ar_event=0.041,
        car_0_1=0.071,
        car_m1_p1=0.071,
        car_0_3=0.110,
        car_0_5=0.110,
        car_m5_p5=0.105,
        is_rockstar=1,
        event_type_release=1,
        franchise_gta=1
    )
    
    print("🎮 Event Impact Prediction:")
    print(f"   Predicted Impact: {result['impact']}")
    print(f"   Confidence: {result['confidence']}%")
    print(f"   Probabilities: {result['probabilities']}")


