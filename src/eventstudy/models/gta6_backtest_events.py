"""
Filter events_labeled.csv for GTA VI-related events.
Creates gta6_backtest_events.csv for backtesting ML models.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_RAW = BASE_DIR / "data" / "raw"

# Create raw data folder if needed
DATA_RAW.mkdir(parents=True, exist_ok=True)

print(f"BASE_DIR: {BASE_DIR}")
print(f"DATA_PROCESSED: {DATA_PROCESSED}")
print(f"DATA_RAW: {DATA_RAW}\n")


def create_gta6_backtest_events():
    """
    Filter events_labeled.csv for GTA VI-related events.
    Save to gta6_backtest_events.csv
    """
    
    # Load events
    events_path = DATA_PROCESSED / "events_labeled.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"File not found: {events_path}")
    
    df = pd.read_csv(events_path, sep=";")
    print(f"✅ Loaded events_labeled.csv: {df.shape}\n")
    print(f"   Columns: {df.columns.tolist()}\n")
    
    # Filter for GTA VI-related events
    # Criteria: franchise contains "GTA", game contains "GTA VI" or "GTA6"
    
    mask_gta = df["franchise"].astype(str).str.contains("GTA", case=False, na=False)
    mask_gta6 = (
        df["game"].astype(str).str.contains("GTA VI|GTA6|GTA 6", case=False, na=False) |
        df["event_id"].astype(str).str.contains("GTA6|GTAVI", case=False, na=False)
    )
    
    mask_final = mask_gta & mask_gta6
    
    df_gta6 = df[mask_final].copy()
    
    print(f"🔍 GTA VI-related events found: {len(df_gta6)}\n")
    print("Events selected for backtest:")
    print(df_gta6[["event_id", "trading_date", "game", "event_type", "impact_label", "CAR_m1_p1"]].to_string(index=False))
    
    # Ensure required columns exist
    required_cols = [
        "event_id", "trading_date", "ticker", "is_rockstar", "game", "franchise",
        "event_type", "sentiment", "return", "market_return", "impact_label_num"
    ]
    
    missing = [col for col in required_cols if col not in df_gta6.columns]
    if missing:
        print(f"\n⚠️  Missing columns: {missing}")
    
    # Save to raw data folder
    output_path = DATA_PROCESSED / "gta6_backtest_events.csv"
    df_gta6.to_csv(output_path, sep=";", index=False)
    
    print(f"\n✅ Saved {len(df_gta6)} GTA VI events to: {output_path}")
    print(f"   Shape: {df_gta6.shape}\n")
    
    return df_gta6


if __name__ == "__main__":
    create_gta6_backtest_events()
