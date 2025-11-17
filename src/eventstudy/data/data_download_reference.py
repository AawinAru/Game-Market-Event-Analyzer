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

import kagglehub

# Download latest version
path = kagglehub.dataset_download("patkle/video-game-sales-data-from-vgchartzcom")

print("Path to dataset files:", path)


import kagglehub

# Download latest version
path = kagglehub.dataset_download("anandshaw2001/video-game-sales")

print("Path to dataset files:", path)

