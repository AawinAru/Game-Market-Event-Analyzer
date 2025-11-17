import yfinance as yf

ticker = "TTWO"
start_date = "2010-01-01"
end_date   = "2024-12-31"

# Download (auto_adjust=True by default, so 'Close' is already adj price)
data = yf.download(ticker, start=start_date, end=end_date)

# Flatten multi-index columns if they exist
data.columns = data.columns.droplevel(1) if isinstance(data.columns, tuple) or len(data.columns.names) > 1 else data.columns

# Keep only adjusted close (stored under 'Close')
data = data[["Close"]]

# Rename column to make it clear
data = data.rename(columns={"Close": "Adj Close"})

# Export CSV
data.to_csv("TTWO_2010_2024.csv")

print("✅ File saved as TTWO_2010_2024.csv")
print(data.head())


import yfinance as yf
import pandas as pd

tickers = ["EA", "ATVI", "UBI.PA", "NTDOY", "^GSPC"]
start_date = "2015-01-01"
end_date   = "2024-12-31"

# Download all tickers
data = yf.download(tickers, start=start_date, end=end_date)

# Flatten multi-index columns if they exist
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Keep only adjusted close prices (stored under 'Close')
data = data[["Close"]].copy()

# Rename column to avoid confusion
data = data.rename(columns={"Close": "Adj Close"})

# If multiple tickers were downloaded, "Adj Close" is now a single column name repeated
# We need to restore each ticker name from the original dataset:
data.columns = tickers  # assign ticker names as columns

# Export to CSV
data.to_csv("GameStocks_SP500_2015_2024.csv")

print("✅ CSV saved: GameStocks_SP500_2015_2024.csv")
print(data.head())



import yfinance as yf

ticker = "EA"
start = "2015-01-01"
end   = "2024-12-31"

# Download data (adjusted close is stored in 'Close' because auto_adjust=True by default)
df = yf.download(ticker, start=start, end=end)

# Keep only the adjusted close
df = df[["Close"]]

# Rename for clarity
df = df.rename(columns={"Close": "Adj Close"})

# Save to CSV
df.to_csv("EA_2015_2024.csv")

print("✅ Saved as EA_2015_2024.csv")
print(df.head())


# requirements: pip install yfinance pandas
import yfinance as yf
import pandas as pd

ticker = "^VIX"
start_date = "2015-01-01"
end_date   = "2024-12-31"
outfile = "VIX_2015_2024.csv"

# Download
df = yf.download(ticker, start=start_date, end=end_date, progress=False)

# Flatten MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# VIX daily value is in 'Close' (yfinance may auto-adjust; for indices Close is the actual index value)
# Keep only Close and rename to "VIX"
if "Close" in df.columns:
    out = df[["Close"]].rename(columns={"Close": "VIX"})
else:
    # fallback: if the returned frame only has one column (rare), use it
    if df.shape[1] == 1:
        out = df.copy()
        out.columns = ["VIX"]
    else:
        raise RuntimeError(f"Could not find 'Close' column in VIX data. Available columns: {list(df.columns)}")

# Ensure Date is the index and sorted
out.index.name = "Date"
out = out.sort_index()

# Save CSV
out.to_csv(outfile, index=True)

# Quick diagnostics
print(f"Saved: {outfile}")
print("Rows:", len(out))
print("Head:")
print(out.head())
print("Tail:")
print(out.tail())


import kagglehub

# Download latest version
path = kagglehub.dataset_download("patkle/video-game-sales-data-from-vgchartzcom")

print("Path to dataset files:", path)


import kagglehub

# Download latest version
path = kagglehub.dataset_download("anandshaw2001/video-game-sales")

print("Path to dataset files:", path)
