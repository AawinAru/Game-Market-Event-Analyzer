"""
Compute daily returns and market returns.
"""

import pandas as pd
from pathlib import Path

# Use __file__ for scripts (not Path.cwd())
BASE_DIR = Path(__file__).resolve().parents[3]

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

print(f"Project root: {BASE_DIR}")
print(f"Data processed: {DATA_PROCESSED}")

# Load prices
prices = pd.read_csv(DATA_PROCESSED / "prices_long.csv")
prices["date"] = pd.to_datetime(prices["date"])
prices["ticker"] = prices["ticker"].astype(str).str.upper()

print(f"\n✅ Loaded {len(prices)} records")
print(prices.head())

# Clean tickers
prices["ticker"] = (
    prices["ticker"]
    .str.replace("^GSPC", "SP500", regex=False)
    .str.replace("UBI.PA", "UBSFY", regex=False)
    .str.upper()
)

print(f"\nUnique tickers: {prices['ticker'].unique()}")

# Compute returns
prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
prices["return"] = prices.groupby("ticker")["adj_close"].pct_change()

print(f"\n✅ Computed daily returns")
print(prices.head(10))

# Add market returns
market_ticker = "SP500"
market = prices[prices["ticker"] == market_ticker][["date", "return"]].rename(
    columns={"return": "market_return"}
)

prices = prices.merge(market, on="date", how="left")

print(f"\n✅ Added market returns")
print(prices.head(10))

# Save results
output_file = DATA_PROCESSED / "prices_with_returns.csv"
prices.to_csv(output_file, index=False)
print(f"\n✅ Saved to: {output_file}")
print(f"   Shape: {prices.shape}")
print(f"   Columns: {prices.columns.tolist()}")


if __name__ == "__main__":
    pass  # All code runs above