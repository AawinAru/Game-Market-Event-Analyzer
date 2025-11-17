"""
Build merged stock prices from multiple sources.
"""

import pandas as pd
from pathlib import Path

# Get the project root directory (go up 3 folders from this file)
BASE_DIR = Path(__file__).resolve().parents[3]

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

print(f"Looking for data in: {DATA_RAW}")
print(f"Data raw exists: {DATA_RAW.exists()}")


def load_ea():
    path = DATA_RAW / "EA_2015_2024.csv"
    df = pd.read_csv(path, skiprows=3, header=None, names=['date', 'adj_close'])
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = "EA"
    return df[["date", "ticker", "adj_close"]]


def load_ttwo():
    """TTWO has normal headers - use them"""
    path = DATA_RAW / "TTWO_2010_2024.csv"
    df = pd.read_csv(path)  # No skiprows! Use the header row
    df = df.rename(columns={"Date": "date", "Adj Close": "adj_close"})
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = "TTWO"
    return df[["date", "ticker", "adj_close"]]


def load_gamestocks():
    """GameStocks has normal headers"""
    path = DATA_RAW / "GameStocks_SP500_2015_2024.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Skip EA column (it's empty), use only: ATVI, UBI.PA, NTDOY, ^GSPC
    long_df = df.melt(
        id_vars=["Date"],
        value_vars=["ATVI", "UBI.PA", "NTDOY", "^GSPC"],  # Removed EA
        var_name="ticker",
        value_name="adj_close",
    )
    long_df = long_df.rename(columns={"Date": "date"})
    return long_df


def build_prices():
    ttwo = load_ttwo()
    ea = load_ea()
    gs = load_gamestocks()

    prices = pd.concat([ttwo, ea, gs], ignore_index=True)

    # clean tickers
    prices["ticker"] = prices["ticker"].str.upper()
    prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Long Format
    prices_long = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_long = DATA_PROCESSED / "prices_long.csv"
    prices_long.to_csv(out_long, index=False)
    print(f"✅ Saved long format: {out_long}")
    print(f"   Shape: {prices_long.shape}")
    print(f"   Columns: {prices_long.columns.tolist()}")

    # Convert from long to wide format
    prices_wide = prices.pivot_table(
        index="date",
        columns="ticker",
        values="adj_close",
        aggfunc="first"  # In case of duplicates, take the first value
    )
    
    # Reset index to make date a column
    prices_wide = prices_wide.reset_index()
    
    # Sort by date
    prices_wide = prices_wide.sort_values("date").reset_index(drop=True)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "prices_wide.csv"
    prices_wide.to_csv(out, index=False)
    print(f"✅ Saved: {out}")


if __name__ == "__main__":
    build_prices()