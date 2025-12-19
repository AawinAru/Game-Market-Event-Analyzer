"""
Backtest OLS regression (Model 2 – with GTA trend z-scores)
on GTA VI-related HOLD-OUT events (not in training).

Uses:
- data/raw/backtest.csv (DELAY1 only)
- results/02_ols_regression/ols_regression_coeffs.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_RAW = BASE_DIR / "data" / "raw"
RESULTS_OLS = BASE_DIR / "results" / "02_ols_regression"
RESULTS_SUMMARY = BASE_DIR / "results" / "05_summary"
RESULTS_SUMMARY.mkdir(parents=True, exist_ok=True)

# ✅ TEST ONLY ON HOLD-OUT EVENT
HOLDOUT_EVENTS = [
    "TTWO_2025_GTA6_DELAY1",  
]


def load_coeffs_model2():
    """Load OLS coefficients for Model 2 from CSV."""
    coeffs = pd.read_csv(RESULTS_OLS / "ols_regression_coeffs.csv")
    coeffs = coeffs[["parameter", "model2_all_trends"]].set_index("parameter")
    beta = coeffs["model2_all_trends"]
    print("\n📐 Loaded Model 2 coefficients:")
    print(beta)
    return beta


def add_dummies_like_ols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate sentiment and event-type dummies used in Model 2:
      sent_negative, sent_positive,
      evt_delay, evt_earnings, evt_leak,
      evt_major announcement, evt_release, evt_trailer/reveal
    """
    df = df.copy()

    # sentiment → lower
    df["sentiment"] = (
        df["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df["sent_negative"] = (df["sentiment"] == "negative").astype(float)
    df["sent_positive"] = (df["sentiment"] == "positive").astype(float)

    # event_type → lower
    df["event_type"] = (
        df["event_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["evt_delay"] = (df["event_type"] == "delay").astype(float)
    df["evt_earnings"] = (df["event_type"] == "earnings").astype(float)
    df["evt_leak"] = (df["event_type"] == "leak").astype(float)
    df["evt_major announcement"] = (df["event_type"] == "major announcement").astype(float)
    df["evt_release"] = (df["event_type"] == "release").astype(float)
    df["evt_trailer/reveal"] = (df["event_type"] == "trailer/reveal").astype(float)

    return df


def build_X_for_events(df_reg: pd.DataFrame, beta: pd.Series) -> pd.DataFrame:
    """
    Build feature rows X for the selected events, with columns in the
    exact order of beta.index (const, market_return, is_rockstar, dummies, trends…)
    """

    df_reg = df_reg.copy()

    # Make sure we **have all columns** that appear in beta.
    # If a column is missing (e.g. a dummy that never occurs), create it as 0.
    for col in beta.index:
        if col == "const":
            continue
        if col not in df_reg.columns:
            df_reg[col] = 0.0

    # Now build X with same order as beta
    feature_cols = [c for c in beta.index if c != "const"]
    X = df_reg[feature_cols].copy()

    # Add constant column at the front
    X.insert(0, "const", 1.0)

    # ensure numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    print("\n🧱 Design matrix X for backtest:")
    print(X)

    return X


def run_backtest_ols():
    """Backtest on HOLD-OUT events only"""
    
    beta = load_coeffs_model2()

    # Load hold-out backtest data
    df_backtest = pd.read_csv(DATA_RAW / "backtest.csv", sep=";")
    
    print(f"\n📊 Backtest dataset loaded:")
    print(f"   Events in backtest.csv: {df_backtest['event_id'].unique().tolist()}")
    
    # Filter to ONLY hold-out events
    df_backtest = df_backtest[df_backtest["event_id"].isin(HOLDOUT_EVENTS)].copy()
    
    if len(df_backtest) == 0:
        print(f"\n❌ No hold-out events found!")
        print(f"   Expected: {HOLDOUT_EVENTS}")
        return
    
    print(f"\n✅ Testing on {len(df_backtest)} HOLD-OUT event(s):")
    print(df_backtest[["event_id", "event_type", "sentiment", "AR_event"]].to_string())

    # Recreate sentiment + event-type dummies
    df_backtest = add_dummies_like_ols(df_backtest)

    # Build X aligned with Model 2 betas
    X = build_X_for_events(df_backtest, beta)

    # True AR_event from data
    y_true = df_backtest["AR_event"].astype(float).values

    # Align beta with X.columns
    beta_vec = beta[X.columns]

    # predicted AR_event = X * beta
    y_pred = X.values @ beta_vec.values

    # Build result table
    res = df_backtest[["event_id", "date", "event_type", "sentiment", "ticker"]].copy()
    res["AR_event_true"] = y_true
    res["AR_event_pred"] = y_pred
    res["error"] = res["AR_event_pred"] - res["AR_event_true"]
    res["abs_error"] = abs(res["error"])

    out_path = RESULTS_SUMMARY / "gta6_backtest_ols_model2.csv"
    res.to_csv(out_path, index=False)

    print("\n" + "="*80)
    print("📉 GTA6 BACKTEST – OLS MODEL 2 (HOLD-OUT EVENTS ONLY)")
    print("="*80)
    print(res.to_string(index=False))
    
    print(f"\n📊 Prediction Accuracy:")
    print(f"   Mean Absolute Error: {res['abs_error'].mean():.6f}")
    print(f"   R²: {1 - (res['error'].std()**2 / (y_true.std()**2 + 1e-10)):.4f}")
    
    print(f"\n✅ Saved OLS backtest results to: {out_path}\n")


def main():
    print("\n" + "="*80)
    print("🔬 OLS REGRESSION BACKTEST (HOLD-OUT VALIDATION)")
    print("="*80)
    run_backtest_ols()


if __name__ == "__main__":
    main()