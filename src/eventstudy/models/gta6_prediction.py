"""
GTA VI Scenario Predictions using OLS Model 2 (with Trends) + Binary Classification.

Generates:
1. OLS predictions: 3×3 scenarios (market × hype)
2. Binary ML predictions: 3 scenarios (bear/base/bull)
- Results: results/05_summary/gta_scenario_predictions*.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ✅ SETUP PATHS
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_OLS = BASE_DIR / "results" / "02_ols_regression"
RESULTS_SUMMARY = BASE_DIR / "results" / "05_summary"

# Create results folders if they don't exist
RESULTS_OLS.mkdir(parents=True, exist_ok=True)
RESULTS_SUMMARY.mkdir(parents=True, exist_ok=True)

print(f"BASE_DIR:       {BASE_DIR}")
print(f"DATA_PROCESSED: {DATA_PROCESSED}")
print(f"RESULTS_OLS:    {RESULTS_OLS}")
print(f"RESULTS_SUMMARY: {RESULTS_SUMMARY}\n")


# ============================================================================
# PART 1: OLS MODEL PREDICTIONS
# ============================================================================

def load_ols_model(model_name="model2"):
    """Load pre-trained OLS model from pickle file."""
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


def build_ols_scenarios(model, model_name="model2"):
    """Build 3×3 scenario grid (market × hype) for OLS predictions."""
    
    cols = model.model.exog_names
    rows = []

    market_scenarios = {
        "bear": -0.02,   # -2% S&P 500 day
        "base":  0.00,   # flat
        "bull":  0.02,   # +2% S&P 500 day
    }

    trend_scenarios = {
        "low":  -1.0,    # 1 std below avg GTA hype
        "mid":   0.0,    # average GTA hype
        "high":  1.0,    # 1 std above avg GTA hype
    }

    print(f"🔨 Building OLS scenarios for {model_name}...")
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
                row["is_rockstar"] = 1.0

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

            # ✅ GTA TRENDS (z-scored)
            for col in ["trend_gta_z", "trend_gta6_z", "trend_gtavi_z", "trend_rockstar_z"]:
                if col in row:
                    row[col] = t_val

            row["scenario"] = f"{m_name}_{t_name}"
            rows.append(row)

    scen_df = pd.DataFrame(rows).set_index("scenario")
    scen_X = scen_df[cols]
    
    print(f"   Scenario feature matrix shape: {scen_X.shape}\n")
    
    # ✅ PREDICT
    preds = model.predict(scen_X)

    scen_df["pred_AR_event"] = preds
    scen_df["market_scenario"] = [s.split("_")[0] for s in scen_df.index]
    scen_df["trend_scenario"] = [s.split("_")[1] for s in scen_df.index]

    return scen_df


# ============================================================================
# PART 2: BINARY ML MODEL PREDICTIONS
# ============================================================================

def load_ml_dataset():
    """Load ML dataset with features."""
    path = DATA_PROCESSED / "ml_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"ml_dataset.csv not found at: {path}")

    df = pd.read_csv(path, sep=";")
    print(f"📥 Loaded ML dataset: {df.shape}")
    return df


def build_xy_for_binary(df):
    """Build X, y for binary classification."""
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

    # Remove rows with NaN
    mask = X.isna().any(axis=1)
    if mask.sum() > 0:
        X = X[~mask]
        y = y[~mask]

    print(f"📐 Final X shape: {X.shape}, y shape: {y.shape}\n")
    return X, y


def train_binary_model(X, y, random_state=42):
    """Train Logistic Regression on full dataset."""
    
    print("🤖 Training Logistic Regression (binary)…\n")

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
        class_weight="balanced"
    )

    model.fit(X, y)
    
    print("✅ Model trained on full dataset\n")
    return model


def build_binary_scenarios(feature_cols):
    """Build 3 scenarios (bear/base/bull) for binary predictions."""
    
    base = {c: 0.0 for c in feature_cols}

    # Fixed attributes for GTA VI Rockstar announcement
    base["is_rockstar"] = 1.0
    base["sentiment_negative"] = 0.0
    base["franchise_gta"] = 1.0
    base["event_type_marketing"] = 0.0
    base["event_type_negative"] = 0.0
    base["event_type_release"] = 1.0

    scenarios = []

    # Bear market
    s_bear = base.copy()
    s_bear["market_return"] = -0.02
    s_bear["vix_level"] = 35.0
    s_bear["vix_30d_mean"] = 30.0
    s_bear["vix_regime_medium"] = 0.0
    s_bear["vix_regime_high"] = 1.0
    scenarios.append(("bear", s_bear))

    # Base market
    s_base = base.copy()
    s_base["market_return"] = 0.00
    s_base["vix_level"] = 20.0
    s_base["vix_30d_mean"] = 18.0
    s_base["vix_regime_medium"] = 1.0
    s_base["vix_regime_high"] = 0.0
    scenarios.append(("base", s_base))

    # Bull market
    s_bull = base.copy()
    s_bull["market_return"] = 0.02
    s_bull["vix_level"] = 15.0
    s_bull["vix_30d_mean"] = 15.0
    s_bull["vix_regime_medium"] = 0.0
    s_bull["vix_regime_high"] = 1.0
    scenarios.append(("bull", s_bull))

    rows = []
    for name, vec in scenarios:
        row = {c: vec[c] for c in feature_cols}
        row["scenario"] = name
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[feature_cols + ["scenario"]]

    print("✅ Binary scenario feature rows:")
    print(df.to_string(index=False), "\n")

    return df


LABEL_MAP = {0: "Not High", 1: "High"}


def predict_binary_scenarios(model, df_scenarios, feature_cols):
    """Generate binary predictions for scenarios."""
    
    X_scen = df_scenarios[feature_cols]
    y_pred = model.predict(X_scen)
    y_proba = model.predict_proba(X_scen)

    results = []
    for i, scenario in enumerate(df_scenarios["scenario"].values):
        cls = int(y_pred[i])
        probs = y_proba[i]

        results.append({
            "scenario": scenario,
            "predicted_class": cls,
            "predicted_label": LABEL_MAP[cls],
            "proba_not_high": float(probs[0]),
            "proba_high": float(probs[1]),
        })

    return pd.DataFrame(results)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate GTA VI scenario predictions (OLS + Binary ML)."""
    
    print("=" * 80)
    print("🎮 GTA VI SCENARIO PREDICTIONS (OLS + BINARY ML)")
    print("=" * 80 + "\n")
    
    # ========================================================================
    # PART 1: OLS PREDICTIONS
    # ========================================================================
    print("📊 PART 1: OLS MODEL 2 PREDICTIONS (3×3 Grid)")
    print("-" * 80 + "\n")
    
    m2 = load_ols_model(model_name="model2")
    scen_ols = build_ols_scenarios(m2, model_name="model2 (GTA Trends)")
    
    print("\n" + "=" * 80)
    print("📊 OLS SCENARIO PREDICTIONS – MODEL 2")
    print("=" * 80)
    print(scen_ols[["market_scenario", "trend_scenario", "pred_AR_event"]].to_string())
    
    # Save OLS results
    ols_out_path = RESULTS_SUMMARY / "gta_scenario_predictions.csv"
    scen_ols.to_csv(ols_out_path, index=False)
    print(f"\n✅ OLS scenario predictions saved to: {ols_out_path}")
    
    # ========================================================================
    # PART 2: BINARY ML PREDICTIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 PART 2: BINARY ML PREDICTIONS (3 Scenarios)")
    print("-" * 80 + "\n")
    
    # Load and prepare data
    df_ml = load_ml_dataset()
    X, y = build_xy_for_binary(df_ml)
    
    # Train binary model
    binary_model = train_binary_model(X, y)
    
    # Build scenarios
    feature_cols = list(X.columns)
    df_scen_binary = build_binary_scenarios(feature_cols)
    
    # Predict
    df_binary_results = predict_binary_scenarios(binary_model, df_scen_binary, feature_cols)
    
    print("\n" + "=" * 80)
    print("📊 BINARY ML SCENARIO PREDICTIONS")
    print("=" * 80)
    print(df_binary_results.to_string(index=False))
    
    # Save binary results
    binary_out_path = RESULTS_SUMMARY / "gta6_scenario_predictions_binary.csv"
    df_binary_results.to_csv(binary_out_path, index=False)
    print(f"\n✅ Binary scenario predictions saved to: {binary_out_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ BOTH PREDICTIONS COMPLETE")
    print("=" * 80)
    print(f"OLS predictions:    {ols_out_path}")
    print(f"Binary predictions: {binary_out_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

