import pandas as pd
from pathlib import Path

# === PATH SETUP ============================================================= #

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

print("BASE_DIR:", BASE_DIR)
print("DATA_RAW exists:", DATA_RAW.exists())
print("DATA_PROCESSED exists:", DATA_PROCESSED.exists())


# === LOAD EVENTS =========================================================== #

def load_events():
    events_path = DATA_RAW / "events.csv"  # ✅ Already correct
    events = pd.read_csv(events_path, sep=";", encoding="latin-1")
    
    print("Columns:", events.columns.tolist())
    print(f"✅ Loaded {len(events)} events")
    
    # Normalize date and ticker
    events["date"] = pd.to_datetime(events["date"], dayfirst=True, errors="coerce")
    events["ticker"] = events["ticker"].astype(str).str.upper()
    
    # Fix Ubisoft ticker: map empty/UBI.PA to UBSFY
    events.loc[events["publisher"] == "Ubisoft", "ticker"] = "UBSFY"
    events["ticker"] = events["ticker"].str.replace("UBI.PA", "UBSFY", regex=False)
    
    print("Unique tickers in events:", events["ticker"].unique())
    
    return events


# === LOAD PRICES =========================================================== #

def load_prices():
    prices_path = DATA_PROCESSED / "prices_with_returns.csv"
    prices = pd.read_csv(prices_path)
    
    # Convert date to datetime
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    
    print("\nUnique tickers in prices:")
    print(prices["ticker"].unique())
    print(f"UBSFY rows: {len(prices[prices['ticker'] == 'UBSFY'])}")
    
    # Check date range for UBSFY
    ubsfy_prices = prices[prices['ticker'] == 'UBSFY']
    print(f"UBSFY date range: {ubsfy_prices['date'].min()} to {ubsfy_prices['date'].max()}")
    
    return prices


# === MERGE EVENTS WITH NEAREST TRADING DAY ================================== #

def merge_events_with_prices(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    # ✅ Rename date columns FIRST
    events = events.rename(columns={"date": "event_date"})
    prices = prices.rename(columns={"date": "trading_date"})
    
    events_sorted = events.sort_values(["ticker", "event_date"]).reset_index(drop=True)
    prices_sorted = prices.sort_values(["ticker", "trading_date"]).reset_index(drop=True)

    merged_list = []

    for ticker in events_sorted["ticker"].unique():
        e = events_sorted[events_sorted["ticker"] == ticker].copy()
        p = prices_sorted[prices_sorted["ticker"] == ticker].copy()

        if p.empty:
            print(f"[WARN] No price data for ticker {ticker}")
            e["trading_date"] = pd.NaT
            e["adj_close"] = pd.NA
            e["return"] = pd.NA
            e["market_return"] = pd.NA
            merged_list.append(e)
            continue

        tmp = pd.merge_asof(
            e.sort_values("event_date"),
            p.sort_values("trading_date"),
            left_on="event_date",
            right_on="trading_date",
            direction="backward"
        )

        merged_list.append(tmp)

    merged = pd.concat(merged_list, ignore_index=True)
    return merged


# === MAIN FUNCTION =========================================================== #

def main():
    print("Loading events...")
    events = load_events()

    print("Loading prices...")
    prices = load_prices()

    print("Merging...")
    merged = merge_events_with_prices(events, prices)

    out_path = DATA_PROCESSED / "events_with_returns.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()