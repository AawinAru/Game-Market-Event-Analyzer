"""
Backtest ML model on GTA VI–related events.

- Trains the SAME multiclass Gradient Boosting model as train_classification.py
  on impact_label_num (0=Low, 1=Medium, 2=High)
- Uses the SAME feature set as gta6_prediction.py (NO AR_event in features)
- Applies it to gta6_backtest_events.csv
- Compares predicted vs actual impact_label_num
"""

from pathlib import Path
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
# -----------------------------------------------------
# Paths
# -----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_RAW = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "results" / "05_summary"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_PATH = DATA_RAW / "gta6_backtest_events.csv"

# SAME FEATURES AS train_classification.py + gta6_prediction.py (no AR_event)
FEATURE_COLS = [
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


# -----------------------------------------------------
# 1. Load training dataset (ml_dataset.csv)
# -----------------------------------------------------
def load_ml_dataset():
    path = DATA_PROCESSED / "ml_dataset.csv"
    df = pd.read_csv(path, sep=";")
    print(f"✅ Loaded ml_dataset: {df.shape}")
    print("   Columns:", df.columns.tolist())

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in ml_dataset: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df["impact_label_num"].astype(int)

    # Drop NaNs in features/target
    mask = X.isna().any(axis=1) | y.isna()
    if mask.sum() > 0:
        print(f"⚠️ Dropping {mask.sum()} rows with NaNs from ML dataset")
        X = X[~mask].reset_index(drop=True)
        y = y[~mask].reset_index(drop=True)

    print(f"✅ ML training data ready: X={X.shape}, y={y.shape}")
    return X, y


# -----------------------------------------------------
# 2. Load VIX (same logic as build_ml_dataset.py)
# -----------------------------------------------------
def load_vix():
    vix_files = list(DATA_RAW.glob("VIX_*.csv"))
    if not vix_files:
        raise FileNotFoundError(f"No VIX file found in {DATA_RAW}")
    path = vix_files[0]
    print(f"📈 Loading VIX from: {path}")

    vix = pd.read_csv(path)
    vix.columns = [c.lower().strip() for c in vix.columns]

    if "date" not in vix.columns:
        raise KeyError(f"'date' column not found in VIX file. Columns: {vix.columns}")

    vix["date"] = pd.to_datetime(vix["date"])

    # detect level column
    if "^vix" in vix.columns:
        level_col = "^vix"
    elif "adj close" in vix.columns:
        level_col = "adj close"
    elif "close" in vix.columns:
        level_col = "close"
    elif "vix" in vix.columns:
        level_col = "vix"
    else:
        raise KeyError(f"No VIX level column found. Columns: {vix.columns}")

    vix = vix.rename(columns={level_col: "vix_level"})
    vix = vix[["date", "vix_level"]].sort_values("date")

    # 30-day mean
    vix["vix_30d_mean"] = vix["vix_level"].rolling(window=30, min_periods=1).mean()

    # regimes (same idea as build_ml_dataset)
    vix["vix_regime"] = pd.qcut(
        vix["vix_level"],
        q=3,
        labels=["low", "medium", "high"]
    )

    print("📈 VIX sample:\n", vix.head())
    return vix


# -----------------------------------------------------
# 3. Prepare GTA VI backtest features
# -----------------------------------------------------
def simplify_event_type(et: str) -> str:
    et = str(et).strip().lower()

    if "release" in et or "launch" in et:
        return "release"

    if (
        "trailer" in et
        or "reveal" in et
        or "screenshot" in et
        or "gameplay" in et
        or ("announcement" in et and "earn" not in et and "corporate" not in et)
    ):
        return "marketing"

    if (
        "delay" in et
        or "leak" in et
        or "controvers" in et
        or "lawsuit" in et
        or "warning" in et
        or "failure" in et
    ):
        return "negative"

    if "earn" in et or "corporate" in et or "major" in et:
        return "corporate"

    return "other"


def prepare_backtest_features():
    if not BACKTEST_PATH.exists():
        raise FileNotFoundError(f"Backtest file not found: {BACKTEST_PATH}")

    df = pd.read_csv(BACKTEST_PATH, sep=";")
    print(f"✅ Loaded backtest events: {df.shape}")
    print("   Columns:", df.columns.tolist())

    # basic cleaning
    df["trading_date"] = pd.to_datetime(df["trading_date"])

    df["is_rockstar"] = pd.to_numeric(df["is_rockstar"], errors="coerce").fillna(0).astype(int)

    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df["sentiment_negative"] = (df["sentiment"] == "negative").astype(int)

    df["franchise"] = df["franchise"].astype(str).str.strip().str.lower()
    df["franchise_gta"] = (df["franchise"] == "gta").astype(int)

    df["event_type_simple"] = df["event_type"].apply(simplify_event_type)
    df["event_type_marketing"] = (df["event_type_simple"] == "marketing").astype(int)
    df["event_type_negative"] = (df["event_type_simple"] == "negative").astype(int)
    df["event_type_release"] = (df["event_type_simple"] == "release").astype(int)

    # Merge VIX
    vix = load_vix()
    merged = df.merge(
        vix,
        left_on="trading_date",
        right_on="date",
        how="left",
        suffixes=("", "_vix"),
    ).drop(columns=["date"])

    merged["vix_level"] = merged["vix_level"].ffill().bfill()
    merged["vix_30d_mean"] = merged["vix_30d_mean"].ffill().bfill()
    merged["vix_regime"] = merged["vix_regime"].ffill().bfill()

    merged["vix_regime_medium"] = (merged["vix_regime"] == "medium").astype(int)
    merged["vix_regime_high"] = (merged["vix_regime"] == "high").astype(int)

    missing = [c for c in FEATURE_COLS if c not in merged.columns]
    if missing:
        raise KeyError(f"Missing required feature columns in backtest data: {missing}")

    X_bt = merged[FEATURE_COLS].copy()

    if "impact_label_num" not in merged.columns:
        raise KeyError("Column 'impact_label_num' is missing in gta6_backtest_events.csv")

    y_bt = merged["impact_label_num"].astype(int)

    print(f"✅ Backtest features ready: X={X_bt.shape}, y={y_bt.shape}")
    return merged[["event_id", "trading_date"]], X_bt, y_bt


# -----------------------------------------------------
# 4. Run backtest with Gradient Boosting
# -----------------------------------------------------
def run_backtest_gb():
    # 1) Train on full dataset
    X, y = load_ml_dataset()
    ids, X_bt, y_bt = prepare_backtest_features()

    # ---- Sanity check: show NaNs in backtest features ----
    nan_counts = X_bt.isna().sum()
    print("\n🔍 NaNs per feature in backtest X:")
    print(nan_counts[nan_counts > 0])

    # 2) Fit model on CLEAN training data (X has no NaNs already)
    gb = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=300,
        random_state=42,
    )
    gb.fit(X, y)

    # 3) Impute NaNs in backtest features using TRAINING means
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X)  # learn means from training data

    X_bt_clean = pd.DataFrame(
        imputer.transform(X_bt),
        columns=X_bt.columns,
        index=X_bt.index,
    )

    # 4) Predict
    proba = gb.predict_proba(X_bt_clean)
    y_hat = gb.predict(X_bt_clean)

    df_res = ids.copy()
    df_res["true_label"] = y_bt
    df_res["pred_label"] = y_hat
    df_res["proba_low"] = proba[:, 0]
    df_res["proba_medium"] = proba[:, 1]
    df_res["proba_high"] = proba[:, 2]

    label_map = {0: "Low", 1: "Medium", 2: "High"}
    df_res["true_label_str"] = df_res["true_label"].map(label_map)
    df_res["pred_label_str"] = df_res["pred_label"].map(label_map)

    out_path = RESULTS_DIR / "gta6_backtest_ml_gradient_boosting.csv"
    df_res.to_csv(out_path, index=False)

    print("\n======================================================================")
    print("🎯 GTA6 BACKTEST – GRADIENT BOOSTING (MULTICLASS)")
    print("======================================================================")
    print(df_res)
    print(f"\n✅ Saved ML backtest results to: {out_path}\n")

def main():
    run_backtest_gb()


if __name__ == "__main__":
    main()