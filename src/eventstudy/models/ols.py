"""
OLS regression of abnormal returns on event features and GTA Google Trends.

- Input:  data/processed/regression_dataset_with_gta_trends.csv
- Output:
    - results/02_ols_regression/ols_regression_results.txt  (all 3 models)
    - results/02_ols_regression/ols_regression_coeffs.csv   (all params)
    - results/02_ols_regression/gta_scenario_predictions.csv (bull/base/bear AR_event)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

# ✅ SETUP PATHS
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_OLS = BASE_DIR / "results" / "02_ols_regression"

# Create results folder if it doesn't exist
RESULTS_OLS.mkdir(parents=True, exist_ok=True)

print(f"BASE_DIR:       {BASE_DIR}")
print(f"DATA_PROCESSED: {DATA_PROCESSED}")
print(f"RESULTS_OLS:    {RESULTS_OLS}\n")

# Which return we explain
TARGET_VAR = "AR_event"


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
def load_regression_data():
    """Load regression dataset with GTA trends."""
    path = DATA_PROCESSED / "regression_dataset_with_gta_trends.csv"

    if not path.exists():
        raise FileNotFoundError(f"❌ File not found: {path}")

    print(f"📥 Loading regression dataset from: {path}")
    df = pd.read_csv(path, sep=";")

    # Parse dates if present (not used in OLS, just clean)
    for col in ["event_date", "trading_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    print(f"✅ Loaded regression dataset: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}\n")
    return df


# ---------------------------------------------------------------------
# Feature preparation (cleaner, avoids dummy trap)
# ---------------------------------------------------------------------
def prepare_features(df, target_col=TARGET_VAR, use_trends=True, ttwo_only=False):
    """
    Build X, y for OLS with proper index tracking.
    """

    data = df.copy()

    # Restrict to TTWO if needed
    if ttwo_only:
        before = len(data)
        data = data[data["ticker"] == "TTWO"].copy()
        data = data.reset_index(drop=True)  # ✅ Reset index after filtering
        print(f"🔎 TTWO-only subset: {len(data)} rows (was {before})")

    # Drop rows with missing target
    data = data.dropna(subset=[target_col])
    data = data.reset_index(drop=True)  # ✅ Reset index after dropping NaN
    print(f"✅ Rows after dropping missing {target_col}: {len(data)}")

    # ----------------- Target -----------------
    y = data[target_col].copy()

    # ----------------- Base numeric features -----------------
    X = pd.DataFrame(index=data.index)

    for col in ["market_return", "is_rockstar"]:
        if col not in data.columns:
            raise KeyError(f"Missing required column: {col}")
        X[col] = data[col]

    # ----------------- Sentiment dummies -----------------
    if "sentiment" in data.columns:
        sent = data["sentiment"].astype(str).str.lower().str.strip()
    else:
        sent = pd.Series("", index=data.index)

    # Baseline = neutral → include only negative & positive
    X["sent_negative"] = (sent == "negative").astype(int)
    X["sent_positive"] = (sent == "positive").astype(int)

    print("   Sentiment dummies added: ['sent_negative', 'sent_positive']")

    # ----------------- Event type dummies -----------------
    if "event_type" in data.columns:
        evt = data["event_type"].astype(str).str.lower().str.strip()
    else:
        evt = pd.Series("", index=data.index)

    # Baseline = "corporate event" (all zeros)
    def evt_flag(label):
        return (evt == label).astype(int)

    X["evt_delay"] = evt_flag("delay")
    X["evt_earnings"] = evt_flag("earnings")
    X["evt_leak"] = evt_flag("leak")
    X["evt_major announcement"] = evt_flag("major announcement")
    X["evt_release"] = evt_flag("release")
    X["evt_trailer/reveal"] = evt_flag("trailer/reveal")

    print(
        "   Event-type dummies added: "
        "['evt_delay','evt_earnings','evt_leak',"
        "'evt_major announcement','evt_release','evt_trailer/reveal']"
    )

    # ----------------- GTA trend features (z-scored) -----------------
    trend_cols = []
    if use_trends:
        candidate_trends = [
            "trend_gta_z",
            "trend_gta6_z",
            "trend_gtavi_z",
            "trend_rockstar_z",
        ]
        for col in candidate_trends:
            if col in data.columns:
                X[col] = data[col].fillna(0.0)
                trend_cols.append(col)
        if trend_cols:
            print(f"   Using GTA trend z-features: {trend_cols}")
        else:
            print("⚠️ use_trends=True but no trend z-columns found → running without trends.")

    # ----------------- Numeric conversion & cleaning -----------------
    print("\n🔧 Converting X and y to numeric...")

    for col in X.columns:
        # convert bool→int and everything to numeric
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
        X[col] = pd.to_numeric(X[col], errors="coerce")

    y = pd.to_numeric(y, errors="coerce")

    mask = X.isna().any(axis=1) | y.isna()
    n_dropped = mask.sum()
    if n_dropped > 0:
        print(f"   Dropped {n_dropped} rows with NaN in predictors/target")
        X = X[~mask].copy()
        y = y[~mask].copy()

    # Reset indices
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Add constant after numeric conversion
    X = sm.add_constant(X)

    print(f"📐 Final feature matrix: X={X.shape}, y={y.shape}")
    print(f"   Columns: {list(X.columns)}\n")

    # ✅ FIXED: Return data aligned with X, y (all have same RangeIndex)
    return X, y, data  # data already has matching index

# ---------------------------------------------------------------------
# Leave-One-Out Cross-Validation for OLS
# ---------------------------------------------------------------------
def loo_cv_ols(X, y, name="Model"):
    """
    Naive LOO-CV for OLS:
    - For each i, fit on all observations except i
    - Predict y_i
    - Compute LOO-MAE, LOO-MSE, LOO-R²
    """
    n = len(y)
    if n < 5:
        print(f"⚠️ {name}: too few observations ({n}) for meaningful LOO.")
        return {
            "n": n,
            "loo_mae": np.nan,
            "loo_mse": np.nan,
            "loo_r2": np.nan,
        }

    y_true = y.values.astype(float)
    preds = np.empty(n, dtype=float)

    # ✅ RESET INDICES TO 0, 1, 2, ...
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)

    for i in range(n):
        # train on all but i
        X_train = X_reset.drop(index=i)
        y_train = y_reset.drop(index=i)

        model_i = sm.OLS(y_train, X_train).fit()
        # predict on left-out i
        y_hat_i = model_i.predict(X_reset.iloc[[i]]).values[0]  # ✅ Use .values[0]
        preds[i] = y_hat_i

    errors = y_true - preds
    loo_mse = float(np.mean(errors ** 2))
    loo_mae = float(np.mean(np.abs(errors)))

    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    ssr = float(np.sum(errors ** 2))
    loo_r2 = 1.0 - ssr / sst if sst > 0 else np.nan

    print("\n" + "-" * 70)
    print(f"🔁 LOO-CV RESULTS – {name}")
    print("-" * 70)
    print(f"n           : {n}")
    print(f"LOO-MAE     : {loo_mae:.6f}")
    print(f"LOO-MSE     : {loo_mse:.6f}")
    print(f"LOO-R²      : {loo_r2:.4f}")
    print("-" * 70 + "\n")

    return {
        "n": n,
        "loo_mae": loo_mae,
        "loo_mse": loo_mse,
        "loo_r2": loo_r2,
    }

# ---------------------------------------------------------------------
# Run one OLS model
# ---------------------------------------------------------------------
def run_model(name, df, target_col=TARGET_VAR, use_trends=True, ttwo_only=False):
    X, y, data_used = prepare_features(
        df, target_col=target_col, use_trends=use_trends, ttwo_only=ttwo_only
    )

    model = sm.OLS(y, X).fit()

    print("=" * 80)
    print(name)
    print("=" * 80)
    print(model.summary())
    print("\n")

    return model, data_used, X, y  # ✅ Returns 4 values


# ---------------------------------------------------------------------
# Scenario builder: bull / base / bear for GTA VI-like TTWO event
# ---------------------------------------------------------------------
def build_gta_scenarios(model):
    """
    Build bull/base/bear scenarios for a GTA VI-style event:
    - TTWO
    - is_rockstar = 1
    - positive sentiment
    - 'major announcement' event type
    - Different market returns and GTA trend z-levels.
    """

    cols = model.model.exog_names  # includes 'const'
    rows = []

    market_scenarios = {
        "bear": -0.02,   # -2% SP500 day
        "base":  0.00,   # flat
        "bull":  0.02,   # +2% SP500 day
    }

    trend_scenarios = {
        "low":  -1.0,    # 1 std below avg GTA hype
        "mid":   0.0,    # average GTA hype
        "high":  1.0,    # 1 std above avg GTA hype
    }

    for m_name, m_val in market_scenarios.items():
        for t_name, t_val in trend_scenarios.items():
            row = {c: 0.0 for c in cols}

            if "const" in row:
                row["const"] = 1.0

            # Market & Rockstar
            if "market_return" in row:
                row["market_return"] = m_val
            if "is_rockstar" in row:
                row["is_rockstar"] = 1.0  # GTA VI = Rockstar game

            # Sentiment: assume positive hype
            if "sent_negative" in row:
                row["sent_negative"] = 0.0
            if "sent_positive" in row:
                row["sent_positive"] = 1.0

            # Event type: assume "major announcement"
            for col in row.keys():
                if col.startswith("evt_"):
                    row[col] = 0.0
            if "evt_major announcement" in row:
                row["evt_major announcement"] = 1.0

            # GTA trends: set to trend z-level
            for col in ["trend_gta_z", "trend_gta6_z", "trend_gtavi_z", "trend_rockstar_z"]:
                if col in row:
                    row[col] = t_val

            row["scenario"] = f"{m_name}_{t_name}"
            rows.append(row)

    scen_df = pd.DataFrame(rows).set_index("scenario")

    # Ensure columns order matches model exog
    scen_X = scen_df[cols]
    preds = model.predict(scen_X)

    scen_df["pred_AR_event"] = preds
    scen_df["market_scenario"] = [s.split("_")[0] for s in scen_df.index]
    scen_df["trend_scenario"] = [s.split("_")[1] for s in scen_df.index]

    return scen_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    df = load_regression_data()

    # -------- Model 1: baseline, all publishers, no GTA trends --------
    m1, d1, X1, y1 = run_model(
        "MODEL 1 – Baseline (all publishers, no GTA trends)",
        df,
        target_col=TARGET_VAR,
        use_trends=False,
        ttwo_only=False,
    )
    loo1 = loo_cv_ols(X1, y1, name="MODEL 1 – Baseline")

    # -------- Model 2: all publishers + GTA trends --------
    m2, d2, X2, y2 = run_model(
        "MODEL 2 – All publishers + GTA trend z-features",
        df,
        target_col=TARGET_VAR,
        use_trends=True,
        ttwo_only=False,
    )
    loo2 = loo_cv_ols(X2, y2, name="MODEL 2 – All publishers + GTA trends")

    # -------- Model 3: TTWO only + GTA trends --------
    m3, d3, X3, y3 = run_model(
        "MODEL 3 – TTWO-only + GTA trend z-features",
        df,
        target_col=TARGET_VAR,
        use_trends=True,
        ttwo_only=True,
    )
    loo3 = loo_cv_ols(X3, y3, name="MODEL 3 – TTWO-only + GTA trends")

    # -----------------------------------------------------------------
    # MERGE ALL OLS SUMMARIES INTO ONE FILE
    # ✅ SAVE TO RESULTS FOLDER
    # -----------------------------------------------------------------
    merged_summary_path = RESULTS_OLS / "ols_regression_results.txt"

    print(f"💾 Merging all OLS summaries into: {merged_summary_path}")

    with open(merged_summary_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("OLS REGRESSION RESULTS – ALL MODELS\n")
        f.write("=" * 100 + "\n\n")

        def write_model(name, model, loo_stats=None):
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"{name}\n")
            f.write("=" * 100 + "\n")
            f.write(model.summary().as_text())
            f.write("\n")
            if loo_stats is not None:
                f.write("\n--- Leave-One-Out Cross-Validation (AR_event) ---\n")
                f.write(f"n        : {loo_stats['n']}\n")
                f.write(f"LOO-MAE  : {loo_stats['loo_mae']:.6f}\n")
                f.write(f"LOO-MSE  : {loo_stats['loo_mse']:.6f}\n")
                f.write(f"LOO-R²   : {loo_stats['loo_r2']:.4f}\n")
            f.write("\n\n")

        write_model("MODEL 1 – Baseline (All Publishers, No Trends)", m1, loo1)
        write_model("MODEL 2 – All Publishers + GTA Trends", m2, loo2)
        write_model("MODEL 3 – TTWO Only + GTA Trends", m3, loo3)

    print("✅ All summaries (incl. LOO metrics) saved to:", merged_summary_path)

    # -----------------------------------------------------------------
    # MERGE ALL COEFFICIENTS INTO ONE CSV (UNION OF PARAMETERS)
    # ✅ SAVE TO RESULTS FOLDER
    # -----------------------------------------------------------------
    merged_coeffs_path = RESULTS_OLS / "ols_regression_coeffs.csv"

    all_idx = sorted(set(m1.params.index) | set(m2.params.index) | set(m3.params.index))
    coeffs_df = pd.DataFrame(index=all_idx)
    coeffs_df["model1_baseline"] = m1.params.reindex(all_idx)
    coeffs_df["model2_all_trends"] = m2.params.reindex(all_idx)
    coeffs_df["model3_ttwo_trends"] = m3.params.reindex(all_idx)

    coeffs_df = coeffs_df.reset_index().rename(columns={"index": "parameter"})
    coeffs_df.to_csv(merged_coeffs_path, index=False)
    print("✅ All model coefficients saved to:", merged_coeffs_path)

    # -----------------------------------------------------------------
    # GTA VI BULL / BASE / BEAR SCENARIOS (using Model 2)
    # ✅ SAVE TO RESULTS FOLDER
    # -----------------------------------------------------------------
    scen_df = build_gta_scenarios(m2)
    print("\n📊 Scenario predictions (Model 2 – AR_event):")
    print(scen_df[["market_scenario", "trend_scenario", "pred_AR_event"]])

    scen_out_path = RESULTS_OLS / "gta_scenario_predictions.csv"
    scen_df.to_csv(scen_out_path, index=False)
    print("✅ Scenario predictions saved to:", scen_out_path)

    print("\n🎉 OLS regression + scenarios + LOO completed cleanly.")
    print(f"📁 All results saved to: {RESULTS_OLS}\n")


if __name__ == "__main__":
    main()