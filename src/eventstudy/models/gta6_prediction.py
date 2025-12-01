"""
Scenario prediction for a GTA 6–style announcement using ML.

- Loads ml_dataset.csv
- Trains Gradient Boosting on impact_label_num (0=Low, 1=Medium, 2=High)
- Builds bear / base / bull scenarios for a Rockstar GTA 6 "major announcement"
- Saves scenario predictions to data/processed/gta6_scenario_predictions_ml.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_MULTICLASS = BASE_DIR / "results" / "04_multiclass_classification"

print(f"BASE_DIR:       {BASE_DIR}")
print(f"DATA_PROCESSED: {DATA_PROCESSED}\n")


# ---------------------------------------------------------------------
# Load ML dataset
# ---------------------------------------------------------------------
def load_ml_dataset():
    path = DATA_PROCESSED / "ml_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"ml_dataset.csv not found at: {path}")

    print(f"📥 Loading ML dataset from: {path}")
    # ✅ ADD sep=";" to read semicolon-delimited CSV
    df = pd.read_csv(path, sep=";")

    print(f"✅ Loaded ml_dataset: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}\n")
    return df


# ---------------------------------------------------------------------
# Build X, y for multiclass model (impact_label_num)
# ---------------------------------------------------------------------
def build_xy_for_multiclass(df):
    """
    Build feature matrix X and target y for multiclass classification.

    We mirror the feature set from train_classification.py:
      - is_rockstar
      - sentiment_negative
      - market_return
      - AR_event
      - franchise_gta
      - vix_level
      - vix_30d_mean
      - event_type_marketing
      - event_type_negative
      - event_type_release
      - vix_regime_medium
      - vix_regime_high
    """
    required_cols = [
        "is_rockstar",
        "sentiment_negative",
        "market_return",
        "AR_event",
        "franchise_gta",
        "vix_level",
        "vix_30d_mean",
        "event_type_marketing",
        "event_type_negative",
        "event_type_release",
        "vix_regime_medium",
        "vix_regime_high",
        "impact_label_num",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in ml_dataset: {missing}")

    # Drop rows with missing label
    df_clean = df.dropna(subset=["impact_label_num"]).copy()

    X = df_clean[
        [
            "is_rockstar",
            "sentiment_negative",
            "market_return",
            "AR_event",
            "franchise_gta",
            "vix_level",
            "vix_30d_mean",
            "event_type_marketing",
            "event_type_negative",
            "event_type_release",
            "vix_regime_medium",
            "vix_regime_high",
        ]
    ].copy()

    y = df_clean["impact_label_num"].astype(int)

    # Ensure numeric dtype
    X = X.apply(pd.to_numeric, errors="coerce")
    mask = X.isna().any(axis=1)
    if mask.sum() > 0:
        print(f"⚠️ Dropping {mask.sum()} rows with NaNs in features")
        X = X[~mask]
        y = y[~mask]

    print(f"📐 Final X shape: {X.shape}, y shape: {y.shape}\n")
    return X, y


# ---------------------------------------------------------------------
# Train Gradient Boosting on full dataset (after quick sanity check)
# ---------------------------------------------------------------------
def train_best_model(X, y, random_state=42):
    """
    Use GradientBoostingClassifier as the main scenario model.
    First do a quick train/test split to print a sanity report,
    then refit on the full dataset for scenario prediction.
    """
    print("🤖 Training Gradient Boosting (sanity check with train/test split)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    gb = GradientBoostingClassifier(random_state=random_state)
    gb.fit(X_train, y_train)

    y_pred = gb.predict(X_test)
    print("📊 Sanity check – Gradient Boosting classification report (hold-out split):")
    print(classification_report(y_test, y_pred))
    print()

    # Refit on full data for final scenario predictions
    print("🔁 Re-training Gradient Boosting on FULL dataset for scenario predictions...")
    gb_full = GradientBoostingClassifier(random_state=random_state)
    gb_full.fit(X, y)

    return gb_full


# ---------------------------------------------------------------------
# Build scenario feature rows for a GTA 6–style announcement event
# ---------------------------------------------------------------------
def build_gta6_scenarios(model_features):
    """
    Build three scenario rows (bear/base/bull) compatible with the ML features.

    Assumptions:
      - Event: GTA 6 major announcement (trailer / key reveal)
      - Rockstar event: is_rockstar = 1
      - GTA franchise: franchise_gta = 1
      - Sentiment: positive → sentiment_negative = 0
      - Event type: "marketing" → event_type_marketing = 1, others 0
      - AR_event is set to 0 for ex-ante prediction (we don't know realized AR yet)
      - VIX regime + market_return vary by scenario
    """

    # Start with a zero vector
    base = {col: 0.0 for col in model_features}

    # Non-scenario-specific features
    base["is_rockstar"] = 1.0
    base["sentiment_negative"] = 0.0      # positive news
    base["AR_event"] = 0.0                # ex-ante prediction
    base["franchise_gta"] = 1.0
    base["event_type_marketing"] = 1.0    # announcement/trailer
    base["event_type_negative"] = 0.0
    base["event_type_release"] = 0.0

    scenarios = []

    # Bear market: negative index, high volatility
    s_bear = base.copy()
    s_bear["market_return"] = -0.02       # -2%
    s_bear["vix_level"] = 35.0
    s_bear["vix_30d_mean"] = 30.0
    s_bear["vix_regime_medium"] = 0.0
    s_bear["vix_regime_high"] = 1.0
    scenarios.append(("bear", s_bear))

    # Base market: flat index, medium volatility
    s_base = base.copy()
    s_base["market_return"] = 0.0
    s_base["vix_level"] = 20.0
    s_base["vix_30d_mean"] = 18.0
    s_base["vix_regime_medium"] = 1.0
    s_base["vix_regime_high"] = 0.0
    scenarios.append(("base", s_base))

    # Bull market: positive index, low volatility
    s_bull = base.copy()
    s_bull["market_return"] = 0.02        # +2%
    s_bull["vix_level"] = 15.0
    s_bull["vix_30d_mean"] = 15.0
    s_bull["vix_regime_medium"] = 0.0
    s_bull["vix_regime_high"] = 0.0
    scenarios.append(("bull", s_bull))

    rows = []
    for name, d in scenarios:
        row = {col: d.get(col, 0.0) for col in model_features}
        row["scenario"] = name
        rows.append(row)

    df_scenarios = pd.DataFrame(rows)
    df_scenarios = df_scenarios[model_features + ["scenario"]]

    print("✅ GTA 6 scenario feature rows (input to model):")
    print(df_scenarios, "\n")

    return df_scenarios


# ---------------------------------------------------------------------
# Map class index to label
# ---------------------------------------------------------------------
IMPACT_MAP = {
    0: "Low impact",
    1: "Medium impact",
    2: "High impact",
}


def run_gta6_scenario_predictions():
    # 1. Load dataset & build X, y
    df_ml = load_ml_dataset()
    X, y = build_xy_for_multiclass(df_ml)

    # 2. Train Gradient Boosting on full dataset
    model = train_best_model(X, y, random_state=42)

    # 3. Build scenarios
    feature_cols = list(X.columns)  # all input features
    df_scen = build_gta6_scenarios(feature_cols)

    X_scen = df_scen[feature_cols].copy()
    scenario_names = df_scen["scenario"].values

    # 4. Predict class + probabilities
    y_pred = model.predict(X_scen)
    y_proba = model.predict_proba(X_scen)

    results = []
    for i, sc in enumerate(scenario_names):
        cls = int(y_pred[i])
        label = IMPACT_MAP.get(cls, f"class_{cls}")
        proba_vec = y_proba[i]

        results.append(
            {
                "scenario": sc,
                "predicted_class": cls,
                "predicted_label": label,
                "proba_low": float(proba_vec[0]),
                "proba_medium": float(proba_vec[1]),
                "proba_high": float(proba_vec[2]),
            }
        )

    df_results = pd.DataFrame(results)

    # 5. Save results
    out_path = RESULTS_MULTICLASS / "gta6_scenario_predictions_ml.csv"
    df_results.to_csv(out_path, index=False)

    print("📊 GTA 6 scenario ML predictions:")
    print(df_results, "\n")
    print(f"✅ Saved GTA 6 scenario predictions to: {out_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_gta6_scenario_predictions()