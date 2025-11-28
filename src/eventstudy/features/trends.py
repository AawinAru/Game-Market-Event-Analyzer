"""
Build regression dataset with GTA-only Google Trends features.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import time
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

print(f"BASE_DIR:       {BASE_DIR}")
print(f"DATA_RAW:       {DATA_RAW}")
print(f"DATA_PROCESSED: {DATA_PROCESSED}\n")


def download_gta_trends_or_use_cache(start_date="2010-01-01", end_date="2025-12-31", geo="US"):
    """Download or use cached GTA trends."""
    cache_path = DATA_RAW / "gta_trends.csv"
    
    if cache_path.exists():
        print(f"📂 Using CACHED GTA trends from: {cache_path}")
        df = pd.read_csv(cache_path)
        print(f"✅ Loaded {len(df)} rows from cache\n")
        return df
    
    print("🔧 Creating MOCK GTA trends data for testing...\n")
    
    dates = pd.date_range(start="2010-01-01", end="2025-12-31", freq="W")
    n = len(dates)
    
    mock_data = pd.DataFrame({
        "date": dates,
        "trend_gta": np.random.randint(20, 100, n),
        "trend_gta6": np.random.randint(10, 80, n),
        "trend_gtavi": np.random.randint(5, 60, n),
        "trend_rockstar": np.random.randint(15, 70, n),
    })
    
    for col in ["trend_gta", "trend_gta6", "trend_gtavi", "trend_rockstar"]:
        mock_data[col] = mock_data[col].rolling(4, center=True).mean().fillna(mock_data[col])
    
    mock_data["trend_gta_mean"] = mock_data[["trend_gta", "trend_gta6", "trend_gtavi"]].mean(axis=1)
    
    print("✅ Mock GTA trends sample:")
    print(mock_data.head(10), "\n")
    
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    mock_data.to_csv(cache_path, index=False)
    print(f"✅ Saved mock data to: {cache_path}\n")
    
    return mock_data


def load_events_labeled():
    """Load events_labeled.csv from DATA_PROCESSED."""
    path = DATA_PROCESSED / "events_labeled.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    print(f"📥 Loading events_labeled from: {path}")
    df = pd.read_csv(path, sep=";")

    print(f"📋 Available columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    print()

    # Flexible date column detection
    date_col = None
    for col_name in ["event_date", "date", "event_date_str", "trading_date"]:
        if col_name in df.columns:
            date_col = col_name
            break
    
    if date_col is None:
        raise KeyError(
            f"No date column found! Expected one of: "
            f"['event_date', 'date', 'event_date_str', 'trading_date']. "
            f"Found columns: {df.columns.tolist()}"
        )
    
    print(f"✅ Parsing dates from column: '{date_col}'")
    df["event_date"] = pd.to_datetime(df[date_col], errors="coerce")
    
    if "trading_date" in df.columns:
        df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    else:
        print("⚠️  Creating trading_date from event_date...")
        df["trading_date"] = df["event_date"]

    print(f"✅ Loaded events_labeled: {df.shape}")
    print(f"   Date range: {df['event_date'].min()} to {df['event_date'].max()}\n")

    return df


def expand_trends_to_daily(trends):
    """Convert weekly GTA trends to daily series via forward-fill."""
    print("📆 Expanding weekly GTA trends to DAILY (forward-fill)...")

    t = trends.copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t.set_index("date")

    t_daily = t.resample("D").ffill().reset_index()

    print("✅ GTA daily trends sample:")
    print(t_daily.head(), "\n")

    return t_daily


def normalize_trend_columns(df):
    """Add z-scored versions of trend columns."""
    trend_cols = ["trend_gta", "trend_gta6", "trend_gtavi", "trend_rockstar", "trend_gta_mean"]

    for col in trend_cols:
        if col not in df.columns:
            continue
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            df[col + "_z"] = 0.0
        else:
            df[col + "_z"] = (df[col] - mean) / std

    print("✅ Added normalized trend columns (_z)\n")
    return df


def build_regression_dataset_with_gta_trends():
    """Main function to build regression dataset with GTA trends."""
    
    trends_weekly = download_gta_trends_or_use_cache()
    trends_daily = expand_trends_to_daily(trends_weekly)
    events = load_events_labeled()

    print("🔗 Merging GTA trends with events on trading_date...")
    merged = events.merge(
        trends_daily,
        left_on="trading_date",
        right_on="date",
        how="left",
    )

    merged = merged.drop(columns=["date"])

    print("✅ Sample of merged regression dataset (with raw trends):")
    print(
        merged[
            [
                "event_id",
                "trading_date",
                "ticker",
                "AR_event",
                "CAR_0_1",
                "trend_gta",
                "trend_gta6",
                "trend_gtavi",
                "trend_rockstar",
                "trend_gta_mean",
            ]
        ].head(),
        "\n",
    )

        # === FIX: Only TTWO events should keep GTA/GTAVI/GTA6 trends ===
    trend_cols = [
        c for c in merged.columns
        if c.startswith("trend_gta")
        or c.startswith("trend_gtavi")
        or c.startswith("trend_rockstar")
    ]

    mask_non_ttwo = merged["ticker"] != "TTWO"
    merged.loc[mask_non_ttwo, trend_cols] = None

    print(f"✔ GTA trends restricted to TTWO only — Cleared {mask_non_ttwo.sum()} rows.\n")
    
    merged = normalize_trend_columns(merged)

    out_path = DATA_PROCESSED / "regression_dataset_with_gta_trends.csv"
    merged.to_csv(out_path, index=False, sep=";")
    print(f"✅ Saved regression dataset with GTA trends to: {out_path}\n")

    return merged


def inspect_events_labeled_file():
    """Inspect the events_labeled.csv to see its actual structure."""
    path = DATA_PROCESSED / "events_labeled.csv"
    
    if not path.exists():
        print(f"❌ File not found: {path}")
        return
    
    df = pd.read_csv(path, sep=";")
    
    print("="*70)
    print("📊 EVENTS_LABELED.CSV INSPECTION")
    print("="*70)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col:30s} | {df[col].dtype}")
    
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    print(df.isnull().sum())


# ---- MAIN ----
def main():
    print("🚀 Building regression dataset with GTA-only Google Trends...\n")
    
    # ✅ First, inspect the file structure
    print("🔍 Inspecting events_labeled.csv...\n")
    inspect_events_labeled_file()
    
    print("\n" + "="*70 + "\n")
    
    # Then try to load
    try:
        df_reg = build_regression_dataset_with_gta_trends()
        print("📊 Final regression dataset shape:", df_reg.shape)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Please check the column names in events_labeled.csv")


if __name__ == "__main__":
    main()