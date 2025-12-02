"""
Backtest OLS regression (Model 2 – with GTA trend z-scores)
on GTA VI-related events.

Uses:
- data/processed/regression_dataset_with_gta_trends.csv
- results/02_ols_regression/ols_regression_coeffs.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_OLS = BASE_DIR / "results" / "02_ols_regression"
RESULTS_SUMMARY = BASE_DIR / "results" / "05_summary"
RESULTS_SUMMARY.mkdir(parents=True, exist_ok=True)

EVENT_IDS = [
    "TTWO_2025_GTA6_DELAY1",
    "TTWO_2025_GTA6_TRAILER2",
    "TTWO_2025_GTA6_DELAY2",
    "GTA6_2022_LEAK",
    "TTWO_2023_GTA6_TRAILER1",
]


# ---------------------------------------------------------------------
# Load Model 2 coefficients
# ---------------------------------------------------------------------
def load_coeffs_model2():
    """Load OLS coefficients for Model 2 from CSV."""
    coeffs = pd.read_csv(RESULTS_OLS / "ols_regression_coeffs.csv")
    # columns: parameter, model1_baseline, model2_all_trends, model3_ttwo_trends
    coeffs = coeffs[["parameter", "model2_all_trends"]].set_index("parameter")
    beta = coeffs["model2_all_trends"]
    print("\n📐 Loaded Model 2 coefficients:")
    print(beta)
    return beta


# ---------------------------------------------------------------------
# Recreate dummies exactly like in ols.py
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Build X matrix aligned with beta (Model 2)
# ---------------------------------------------------------------------
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

    print("\n🧱 Design matrix X for backtest (first rows):")
    print(X.head())

    return X


# ---------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------
def run_backtest_ols():
    beta = load_coeffs_model2()

    # Load regression dataset with trends
    df_reg = pd.read_csv(
        DATA_PROCESSED / "regression_dataset_with_gta_trends.csv",
        sep=";",
    )

    # Filter GTA VI-related events
    df_reg = df_reg[df_reg["event_id"].isin(EVENT_IDS)].copy()
    print(f"\n✅ Found {len(df_reg)} GTA6 events in regression dataset")

    # Recreate sentiment + event-type dummies
    df_reg = add_dummies_like_ols(df_reg)

    # Build X aligned with Model 2 betas
    X = build_X_for_events(df_reg, beta)

    # True AR_event from data
    y_true = df_reg["AR_event"].astype(float).values

    # Align beta with X.columns
    beta_vec = beta[X.columns]   # includes 'const' + all features in same order

    # predicted AR_event = X * beta
    y_pred = X.values @ beta_vec.values

    # Build result table
    res = df_reg[["event_id", "event_date", "event_type", "sentiment"]].copy()
    res["AR_event_true"] = y_true
    res["AR_event_pred"] = y_pred
    res["error"] = res["AR_event_pred"] - res["AR_event_true"]

    out_path = RESULTS_SUMMARY / "gta6_backtest_ols_model2.csv"
    res.to_csv(out_path, index=False)

    print("\n======================================================================")
    print("📉 GTA6 BACKTEST – OLS MODEL 2 (AR_event)")
    print("======================================================================")
    print(res)
    print(f"\n✅ Saved OLS backtest results to: {out_path}\n")


def main():
    run_backtest_ols()


if __name__ == "__main__":
    main()