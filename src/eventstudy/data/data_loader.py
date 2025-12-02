"""
Unified data loading module.
Merges: data_download_reference.py + build_prices.py
- Downloads raw data from Yahoo Finance
- Builds price datasets
- Saves to data/processed/
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

# ---- Setup paths ----
BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DATA = BASE_DIR / "data" / "raw"
PROCESSED_DATA = BASE_DIR / "data" / "processed"

RAW_DATA.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

print(f"BASE_DIR: {BASE_DIR}")
print(f"RAW_DATA: {RAW_DATA}")
print(f"PROCESSED_DATA: {PROCESSED_DATA}\n")


# =====================================================================
# PART 1: DOWNLOAD RAW DATA
# =====================================================================

def download_all_data():
    """Download all raw data from Yahoo Finance"""
    
    print("📊 Downloading TTWO...")
    data = yf.download("TTWO", start="2010-01-01", end="2025-11-15", progress=False)
    data = data[["Close"]].rename(columns={"Close": "Adj Close"})
    data.to_csv(RAW_DATA / "TTWO_2010_2025.csv", index=True)
    print(f"   ✅ Saved: TTWO_2010_2025.csv\n")

    print("📊 Downloading game stocks...")
    tickers = ["EA", "ATVI", "UBSFY", "NTDOY", "^GSPC"]
    data = yf.download(tickers, start="2010-01-01", end="2025-11-15", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    data.columns = tickers
    data.to_csv(RAW_DATA / "GameStocks_SP500_2010_2025.csv", index=True)
    print(f"   ✅ Saved: GameStocks_SP500_2010_2025.csv\n")

    print("📊 Downloading EA...")
    df = yf.download("EA", start="2010-01-01", end="2025-11-15", progress=False)
    df = df[["Close"]].rename(columns={"Close": "Adj Close"})
    df.to_csv(RAW_DATA / "EA_2010_2025.csv", index=True)
    print(f"   ✅ Saved: EA_2010_2025.csv\n")

    print("📊 Downloading VIX...")
    df = yf.download("^VIX", start="2010-01-01", end="2025-11-15", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    else:
        df = df[["Close"]]
    df = df.rename(columns={"Close": "VIX"})
    df.to_csv(RAW_DATA / "VIX_2010_2025.csv", index=True)
    print(f"   ✅ Saved: VIX_2010_2025.csv\n")


# =====================================================================
# PART 2: BUILD PRICES
# =====================================================================

def load_ea():
    """Load EA prices"""
    path = RAW_DATA / "EA_2010_2025.csv"
    df = pd.read_csv(path, skiprows=3, header=None, names=['date', 'adj_close'])
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = "EA"
    return df[["date", "ticker", "adj_close"]]


def load_ttwo():
    """Load TTWO prices"""
    path = RAW_DATA / "TTWO_2010_2025.csv"
    df = pd.read_csv(path, skiprows=3, header=None, names=['date', 'adj_close'])
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = "TTWO"
    return df[["date", "ticker", "adj_close"]]


def load_gamestocks():
    """Load game stocks"""
    path = RAW_DATA / "GameStocks_SP500_2010_2025.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Skip EA column, use only: ATVI, UBSFY, NTDOY, ^GSPC
    long_df = df.melt(
        id_vars=["Date"],
        value_vars=["ATVI", "UBSFY", "NTDOY", "^GSPC"],
        var_name="ticker",
        value_name="adj_close",
    )
    long_df = long_df.rename(columns={"Date": "date"})
    return long_df


def build_prices():
    """Build price dataset from raw data"""
    print("🔨 Building prices_long.csv...")
    
    ttwo = load_ttwo()
    ea = load_ea()
    gs = load_gamestocks()

    prices = pd.concat([ttwo, ea, gs], ignore_index=True)

    # Clean tickers
    prices["ticker"] = prices["ticker"].str.upper()
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    out_long = PROCESSED_DATA / "prices_long.csv"
    prices.to_csv(out_long, index=False)
    print(f"   ✅ Saved: {out_long}")
    print(f"   Shape: {prices.shape}")
    print(f"   Tickers: {sorted(prices['ticker'].unique().tolist())}")
    print(f"   Date range: {prices['date'].min().date()} to {prices['date'].max().date()}\n")

    return prices


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Main execution - download and build all data"""
    
    print("\n" + "="*80)
    print("📥 DATA LOADER - DOWNLOAD & BUILD PRICES")
    print("="*80 + "\n")
    
    print("🌐 STEP 1: DOWNLOAD RAW DATA")
    print("-" * 80)
    download_all_data()
    
    print("🌐 STEP 2: BUILD PRICES")
    print("-" * 80)
    prices_long = build_prices()
    
    print("="*80)
    print("✅ DATA LOADING COMPLETE")
    print("="*80)
    print(f"📁 All data saved to: {PROCESSED_DATA}\n")
    
    return prices_long


if __name__ == "__main__":
    main()