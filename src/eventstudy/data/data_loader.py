#data_download_reference.py

import yfinance as yf
import pandas as pd
from pathlib import Path

# Setup path
BASE_DIR = Path(__file__).resolve().parents[3]
data_dir = BASE_DIR / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)

print(f"📁 Saving to: {data_dir}\n")

# ============================================================
# TTWO
# ============================================================
print("Downloading TTWO...")
data = yf.download("TTWO", start="2010-01-01", end="2025-11-15", progress=False)
data = data[["Close"]].rename(columns={"Close": "Adj Close"})
data.to_csv(data_dir / "TTWO_2010_2025.csv", index=True)
print(f"✅ Saved: TTWO_2010_2025.csv\n")

# ============================================================
# Game Stocks
# ============================================================
print("Downloading game stocks...")
tickers = ["EA", "ATVI", "UBSFY", "NTDOY", "^GSPC"]
data = yf.download(tickers, start="2010-01-01", end="2025-11-15", progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data = data["Close"]
data.columns = tickers
data.to_csv(data_dir / "GameStocks_SP500_2010_2025.csv", index=True)
print(f"✅ Saved: GameStocks_SP500_2010_2025.csv\n")

# ============================================================
# EA
# ============================================================
print("Downloading EA...")
df = yf.download("EA", start="2010-01-01", end="2025-11-15", progress=False)
df = df[["Close"]].rename(columns={"Close": "Adj Close"})
df.to_csv(data_dir / "EA_2010_2025.csv", index=True)
print(f"✅ Saved: EA_2010_2025.csv\n")

# ============================================================
# VIX
# ============================================================
print("Downloading VIX...")
df = yf.download("^VIX", start="2010-01-01", end="2025-11-15", progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df = df["Close"]
else:
    df = df[["Close"]]
df = df.rename(columns={"Close": "VIX"})
df.to_csv(data_dir / "VIX_2010_2025.csv", index=True)
print(f"✅ Saved: VIX_2010_2025.csv\n")

print("🎉 All data downloaded to data/raw/")

#build_prices.py
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
    path = DATA_RAW / "EA_2010_2025.csv"
    df = pd.read_csv(path, skiprows=3, header=None, names=['date', 'adj_close'])
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = "EA"
    return df[["date", "ticker", "adj_close"]]


def load_ttwo():
    """TTWO has normal headers - use them"""
    path = DATA_RAW / "TTWO_2010_2025.csv"
    df = pd.read_csv(path, skiprows=3, header=None, names=['date', 'adj_close'])
    df = df.rename(columns={"Date": "date", "Adj Close": "adj_close"})
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = "TTWO"
    return df[["date", "ticker", "adj_close"]]


def load_gamestocks():
    """GameStocks has normal headers"""
    path = DATA_RAW / "GameStocks_SP500_2010_2025.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Skip EA column (it's empty), use only: ATVI, UBSFY, NTDOY, ^GSPC
    long_df = df.melt(
        id_vars=["Date"],
        value_vars=["ATVI", "UBSFY", "NTDOY", "^GSPC"],  # Removed EA
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

   


if __name__ == "__main__":
    build_prices()