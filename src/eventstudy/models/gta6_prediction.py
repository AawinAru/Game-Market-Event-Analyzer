"""
GTA VI Scenario Predictions using OLS Model 1 (Baseline).

Loads pre-trained OLS model from ols.py and generates:
- Bull / Base / Bear market scenarios
- Low / Mid / High GTA hype levels
- Results: results/02_ols_regression/gta_scenario_predictions.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# ✅ SETUP PATHS
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_OLS = BASE_DIR / "results" / "02_ols_regression"
RESULTS_SUMMARY = BASE_DIR / "results" / "05_summary"

# Create results folder if it doesn't exist
RESULTS_OLS.mkdir(parents=True, exist_ok=True)

print(f"BASE_DIR:       {BASE_DIR}")
print(f"DATA_PROCESSED: {DATA_PROCESSED}")
print(f"RESULTS_OLS:    {RESULTS_OLS}\n")


# ============================================================================
# LOAD OLS MODEL
# ============================================================================
def load_ols_model(model_name="model1"):
    """
    Load pre-trained OLS model from pickle file.
    
    Args:
        model_name: "model1" (baseline), "model2" (all+trends), or "model3" (TTWO+trends)
    
    Returns:
        Fitted OLS model
    """
    models_path = RESULTS_OLS / "ols_models.pkl"
    
    if not models_path.exists():
        raise FileNotFoundError(
            f"❌ OLS models not found at: {models_path}\n"
            f"   Please run: python src/eventstudy/models/ols.py first"
        )
    
    with open(models_path, "rb") as f:
        models_dict = pickle.load(f)
    
    if model_name not in models_dict:
        raise KeyError(
            f"❌ Model '{model_name}' not found.\n"
            f"   Available: {list(models_dict.keys())}"
        )
    
    model = models_dict[model_name]
    print(f"✅ Loaded {model_name} from: {models_path}")
    print(f"   Model type: {type(model)}")
    print(f"   Parameters: {len(model.params)}\n")
    
    return model


# ============================================================================
# BUILD GTA VI SCENARIOS
# ============================================================================
def build_gta_scenarios(model, model_name="model1"):
    """
    Build bull/base/bear scenarios for a GTA VI-style event:
    - TTWO (Rockstar Games)
    - is_rockstar = 1
    - positive sentiment
    - 'major announcement' event type
    - Different market returns and GTA trend z-levels.
    
    Args:
        model: Fitted OLS model
        model_name: Name of model (for feature validation)
    
    Returns:
        DataFrame with scenario predictions
    """

    cols = model.model.exog_names  # includes 'const'
    rows = []

    market_scenarios = {
        "bear": 0.02,   # -% S&P 500 day
        "base":  0.00,   # flat
        "bull":  -0.02,   # +2% S&P 500 day
    }

    trend_scenarios = {
        "low":  -1.0,    # 1 std below avg GTA hype
        "mid":   0.0,    # average GTA hype
        "high":  2.0,    # 1 std above avg GTA hype
    }

    print(f"🔨 Building scenarios for {model_name}...")
    print(f"   Available features: {cols}\n")

    for m_name, m_val in market_scenarios.items():
        for t_name, t_val in trend_scenarios.items():
            row = {c: 0.0 for c in cols}

            # ✅ CONSTANT
            if "const" in row:
                row["const"] = 1.0

            # ✅ MARKET & ROCKSTAR
            if "market_return" in row:
                row["market_return"] = m_val
            if "is_rockstar" in row:
                row["is_rockstar"] = 1.0  # GTA VI = Rockstar game

            # ✅ SENTIMENT: assume positive hype
            if "sent_negative" in row:
                row["sent_negative"] = 0.0
            if "sent_positive" in row:
                row["sent_positive"] = 1.0

            # ✅ EVENT TYPE: assume "major announcement"
            for col in row.keys():
                if col.startswith("evt_"):
                    row[col] = 0.0
            if "evt_major announcement" in row:
                row["evt_major announcement"] = 1.0

            # ✅ GTA TRENDS (z-scored): set to trend level
            # Only if model includes trends (model2, model3)
            for col in ["trend_gta_z", "trend_gta6_z", "trend_gtavi_z", "trend_rockstar_z"]:
                if col in row:
                    row[col] = t_val

            row["scenario"] = f"{m_name}_{t_name}"
            rows.append(row)

    scen_df = pd.DataFrame(rows).set_index("scenario")

    # ✅ ENSURE COLUMN ORDER MATCHES MODEL
    scen_X = scen_df[cols]
    
    print(f"   Scenario feature matrix shape: {scen_X.shape}")
    print(f"   Sample row (bear_low):\n{scen_X.loc['bear_low']}\n")
    
    # ✅ PREDICT
    preds = model.predict(scen_X)

    scen_df["pred_AR_event"] = preds
    scen_df["market_scenario"] = [s.split("_")[0] for s in scen_df.index]
    scen_df["trend_scenario"] = [s.split("_")[1] for s in scen_df.index]

    return scen_df

# ============================================================================
# MAIN
# ============================================================================
def main():
    """Generate GTA VI scenario predictions using OLS Model 1."""
    
    print("=" * 80)
    print("🎮 GTA VI SCENARIO PREDICTIONS")
    print("=" * 80 + "\n")
    
    # Load baseline model
    m2 = load_ols_model(model_name="model2")
    
    # Build scenarios
    scen_df = build_gta_scenarios(m2, model_name="model2 (GTA Trends)")
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 SCENARIO PREDICTIONS – MODEL 2 (GTA Trends)")
    print("=" * 80)
    print(scen_df[["market_scenario", "trend_scenario", "pred_AR_event"]].to_string())
    
    # Save to CSV
    scen_out_path = RESULTS_SUMMARY / "gta_scenario_predictions.csv"
    scen_df.to_csv(scen_out_path, index=False)
    print(f"\n✅ Scenario predictions saved to: {scen_out_path}")

if __name__ == "__main__":
    main()

"""
GTA 6 scenario prediction using the SAME BINARY model as train_binary.py
Chosen model: LOGISTIC REGRESSION (binary)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_BINARY = BASE_DIR / "results" / "03_binary_classification"

print(f"BASE_DIR:       {BASE_DIR}")
print(f"DATA_PROCESSED: {DATA_PROCESSED}\n")


# ---------------------------------------------------------------------
# Load ML dataset
# ---------------------------------------------------------------------
def load_ml_dataset():
    path = DATA_PROCESSED / "ml_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"ml_dataset.csv not found at: {path}")

    df = pd.read_csv(path, sep=";")
    print(f"📥 Loaded ML dataset: {df.shape}")
    return df


# ---------------------------------------------------------------------
# Build X,y exactly like train_binary.py
# ---------------------------------------------------------------------
def build_xy_for_binary(df):
    required_cols = [
        "is_rockstar",
        "sentiment_negative",
        "market_return",
        "franchise_gta",
        "vix_level",
        "vix_30d_mean",
        "event_type_marketing",
        "event_type_negative",
        "event_type_release",
        "vix_regime_medium",
        "vix_regime_high",
        "impact_high",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in ml_dataset: {missing}")

    df_clean = df.dropna(subset=["impact_high"]).copy()

    X = df_clean[
        [
            "is_rockstar",
            "sentiment_negative",
            "market_return",
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

    y = df_clean["impact_high"].astype(int)

    mask = X.isna().any(axis=1)
    if mask.sum() > 0:
        X = X[~mask]
        y = y[~mask]

    print(f"📐 Final X shape: {X.shape}, y shape: {y.shape}\n")
    return X, y


# ---------------------------------------------------------------------
# Train FINAL model → Logistic Regression (binary)
# ---------------------------------------------------------------------
def train_best_model(X, y, random_state=42):

    print("🤖 Training Logistic Regression (binary, sanity test split)…")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("📊 Classification report (hold-out split):")
    print(classification_report(y_test, y_pred))

    print("🔁 Re-training Logistic Regression on FULL dataset...")
    model.fit(X, y)

    return model


# ---------------------------------------------------------------------
# Build GTA 6 bear / base / bull scenario feature rows
# ---------------------------------------------------------------------
def build_gta6_scenarios(model_features):

    base = {c: 0.0 for c in model_features}

    # fixed attributes for a GTA6 Rockstar announcement
    base["is_rockstar"] = 1.0
    base["sentiment_negative"] = 0.0
    base["franchise_gta"] = 1.0
    base["event_type_marketing"] = 0.0
    base["event_type_negative"] = 0.0
    base["event_type_release"] = 1.0

    scenarios = []

    # Bear
    s_bear = base.copy()
    s_bear["market_return"] = -0.02
    s_bear["vix_level"] = 35.0
    s_bear["vix_30d_mean"] = 30.0
    s_bear["vix_regime_medium"] = 0.0
    s_bear["vix_regime_high"] = 1.0
    scenarios.append(("bear", s_bear))

    # Base
    s_base = base.copy()
    s_base["market_return"] = 0.00
    s_base["vix_level"] = 20.0
    s_base["vix_30d_mean"] = 18.0
    s_base["vix_regime_medium"] = 1.0
    s_base["vix_regime_high"] = 0.0
    scenarios.append(("base", s_base))

    # Bull
    s_bull = base.copy()
    s_bull["market_return"] = 0.02
    s_bull["vix_level"] = 15.0
    s_bull["vix_30d_mean"] = 15.0
    s_bull["vix_regime_medium"] = 0.0
    s_bull["vix_regime_high"] = 1.0
    scenarios.append(("bull", s_bull))

    rows = []
    for name, vec in scenarios:
        row = {c: vec[c] for c in model_features}
        row["scenario"] = name
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[model_features + ["scenario"]]

    print("✅ GTA6 scenario feature rows:")
    print(df, "\n")

    return df


# ---------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------
LABEL_MAP = {0: "Not High impact", 1: "High impact"}


def run_gta6_scenario_predictions():

    df_ml = load_ml_dataset()
    X, y = build_xy_for_binary(df_ml)

    model = train_best_model(X, y)

    feature_cols = list(X.columns)
    df_scen = build_gta6_scenarios(feature_cols)

    X_scen = df_scen[feature_cols]
    scen_names = df_scen["scenario"].values

    y_pred = model.predict(X_scen)
    y_proba = model.predict_proba(X_scen)

    results = []
    for i, scenario in enumerate(scen_names):
        cls = int(y_pred[i])
        probs = y_proba[i]

        results.append({
            "scenario": scenario,
            "predicted_class": cls,
            "predicted_label": LABEL_MAP[cls],
            "proba_not_high": float(probs[0]),
            "proba_high": float(probs[1]),
        })

    df_results = pd.DataFrame(results)

    out_path = RESULTS_SUMMARY / "gta6_scenario_predictions_binary.csv"
    df_results.to_csv(out_path, index=False)

    print("📊 GTA6 Scenario Predictions (Binary):")
    print(df_results)
    print(f"\n✅ Saved to {out_path}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_gta6_scenario_predictions()

