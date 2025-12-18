# merge_event_returns.py

"""
Merge events with price returns on nearest trading day.
"""

import pandas as pd
from pathlib import Path

def merge_event_returns():
    """Merge events with price returns"""
    
    BASE_DIR = Path(__file__).resolve().parents[3]
    DATA_RAW = BASE_DIR / "data" / "raw"
    DATA_PROCESSED = BASE_DIR / "data" / "processed"

    print(f"Project root: {BASE_DIR}")
    print(f"Data raw: {DATA_RAW}")
    print(f"Data processed: {DATA_PROCESSED}")

    # Load events
    events = pd.read_csv(DATA_RAW / "events.csv", sep=";", encoding="utf-8-sig")
    events.columns = events.columns.str.strip()
    events["date"] = pd.to_datetime(events["date"], dayfirst=True, errors="coerce")
    events["ticker"] = events["ticker"].astype(str).str.upper()

    print(f"\n✅ Loaded {len(events)} events")
    print(events.head())

    # Load prices
    prices = pd.read_csv(DATA_PROCESSED / "prices_with_returns.csv")
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper()

    print(f"\n✅ Loaded {len(prices)} price records")

    # Sort prices by ticker and date for forward fill logic
    prices = prices.sort_values(["ticker", "date"])

    # Merge events with prices
    merged = events.merge(prices, left_on=["ticker", "date"], right_on=["ticker", "date"], how="left")

    # Identify price columns from the prices dataframe (excluding ticker and date)
    price_columns = [col for col in prices.columns if col not in ["ticker", "date"]]
    
    # For rows with no match, find the next available trading day
    # Check if any price column is missing (NaN)
    missing_mask = merged[price_columns[0]].isna() if price_columns else pd.Series([False] * len(merged))
    
    if missing_mask.any():
        print(f"\n⚠️  {missing_mask.sum()} events have no exact date match, finding next trading day...")
        
        for idx in merged[missing_mask].index:
            ticker = merged.loc[idx, "ticker"]
            event_date = merged.loc[idx, "date"]
            
            # Find next available trading day for this ticker
            next_trading = prices[(prices["ticker"] == ticker) & (prices["date"] > event_date)]
            
            if not next_trading.empty:
                next_row = next_trading.iloc[0]
                # Update the merged dataframe with next trading day data
                for col in price_columns:
                    merged.loc[idx, col] = next_row[col]
                merged.loc[idx, "date"] = next_row["date"]

    print(f"\n✅ Merged events with prices")
    print(merged.head())

    # Save results
    out_path = DATA_PROCESSED / "events_with_returns.csv"
    merged.to_csv(out_path, sep=";", index=False)
    print(f"\n✅ Saved to: {out_path}")
    print(f"   Shape: {merged.shape}")
    print(f"   Columns: {merged.columns.tolist()}")
    
    return merged


if __name__ == "__main__":
    merge_event_returns()
