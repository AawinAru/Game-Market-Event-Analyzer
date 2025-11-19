"""
Compute Abnormal Returns (AR) and Cumulative Abnormal Returns (CAR)
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
from pathlib import Path

# === SETUP ================================================================= #

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

print(f"BASE_DIR: {BASE_DIR}")
print(f"DATA_PROCESSED exists: {DATA_PROCESSED.exists()}")


# === LOAD DATA ============================================================= #

def load_prices():
    prices = pd.read_csv(DATA_PROCESSED / "prices_with_returns.csv")
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    return prices


def load_events():
    events = pd.read_csv(DATA_PROCESSED / "events_with_returns.csv", sep=";")  # ✅ ADD sep=";"
    print("Columns in events_with_returns.csv:", events.columns.tolist())
    print(events.head())
    
    events["event_date"] = pd.to_datetime(events["event_date"])
    events["trading_date"] = pd.to_datetime(events["trading_date"])
    
    # ✅ Fix: use the correct column name
    if "ticker" in events.columns:
        events["ticker"] = events["ticker"].astype(str).str.upper()
    elif "ticker_x" in events.columns:
        events["ticker"] = events["ticker_x"].astype(str).str.upper()
    elif "ticker_y" in events.columns:
        events["ticker"] = events["ticker_y"].astype(str).str.upper()
    else:
        print("[ERROR] No ticker column found!")
        print("Available columns:", events.columns.tolist())
    
    return events


# === ESTIMATE ALPHA & BETA ================================================= #

def estimate_alpha_beta(prices, ticker, market_ticker="SP500"):
    """Estimate alpha and beta using numpy"""
    df = prices[prices["ticker"] == ticker].dropna(subset=["return", "market_return"])
    
    if df.empty:
        print(f"[WARN] No data for ticker {ticker}")
        return None, None
    
    X = df["market_return"].values
    y = df["return"].values
    
    # Add constant for alpha
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    # Solve using least squares
    params = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    alpha = params[0]
    beta = params[1]
    
    return alpha, beta


def build_alpha_beta_table(prices):
    """Build table of alpha/beta for all tickers"""
    tickers = prices["ticker"].unique()
    ab_table = {}

    for t in tickers:
        alpha, beta = estimate_alpha_beta(prices, t)
        ab_table[t] = {"alpha": alpha, "beta": beta}
        if alpha is not None:
            print(f"  {t}: alpha={alpha:.4f}, beta={beta:.4f}")
    
    return ab_table


# === COMPUTE ABNORMAL RETURNS ============================================== #

def compute_ar(row, ab_table):
    """Compute abnormal return for a single event"""
    ticker = row["ticker"]
    
    # Handle missing ticker in ab_table
    if ticker not in ab_table or ab_table[ticker]["alpha"] is None:
        return None
    
    alpha = ab_table[ticker]["alpha"]
    beta = ab_table[ticker]["beta"]
    
    # Expected return = alpha + beta * market_return
    expected_return = alpha + beta * row["market_return"]
    
    # Abnormal return = actual return - expected return
    abnormal_return = row["return"] - expected_return
    
    return abnormal_return


def compute_prices_ar(prices, ab_table):
    """Add expected return and AR columns to prices"""
    prices["expected_return"] = prices.apply(
        lambda r: ab_table[r["ticker"]]["alpha"] + ab_table[r["ticker"]]["beta"] * r["market_return"]
        if pd.notnull(r["market_return"]) and ab_table[r["ticker"]]["alpha"] is not None else None,
        axis=1
    )
    prices["AR"] = prices["return"] - prices["expected_return"]
    return prices


# === COMPUTE CUMULATIVE ABNORMAL RETURNS ================================== #

def compute_car(prices, ticker, trading_date, window=(-1, 1)):
    """Compute cumulative abnormal return for a window around event date"""
    start = trading_date + pd.Timedelta(days=window[0])
    end = trading_date + pd.Timedelta(days=window[1])
    
    df = prices[(prices["ticker"] == ticker) &
                (prices["date"] >= start) &
                (prices["date"] <= end)]
    
    car = df["AR"].sum()
    return car


def compute_events_car(events, prices):
    """Add CAR columns to events"""
    # AR at event date
    events["AR_event"] = events.apply(
        lambda r: compute_ar(r, ab_table) if pd.notnull(r["trading_date"]) else None,
        axis=1
    )
    
    # CAR(-1, +1) - 3-day window
    events["CAR_m1_p1"] = events.apply(
        lambda r: compute_car(prices, r["ticker"], r["trading_date"], window=(-1, 1))
        if pd.notnull(r["trading_date"]) else None,
        axis=1
    )
    
    # CAR(-5, +5) - 11-day window
    events["CAR_m5_p5"] = events.apply(
        lambda r: compute_car(prices, r["ticker"], r["trading_date"], window=(-5, 5))
        if pd.notnull(r["trading_date"]) else None,
        axis=1
    )
    
    return events


# === MAIN ================================================================== #

def main():
    print("\n📥 Loading data...")
    prices = load_prices()
    print(f"✅ Loaded {len(prices)} price rows")
    print(f"   Tickers: {prices['ticker'].unique()}")
    
    events = load_events()
    print(f"✅ Loaded {len(events)} events")
    
    print("\n📊 Estimating alpha & beta...")
    global ab_table
    ab_table = build_alpha_beta_table(prices)
    
    print("\n📈 Computing prices AR...")
    prices = compute_prices_ar(prices, ab_table)
    
    print("\n📊 Computing events AR & CAR...")
    events = compute_events_car(events, prices)
    
    print("\n✅ Sample results:")
    print(events[["event_id", "ticker", "event_date", "trading_date", "AR_event", "CAR_m1_p1", "CAR_m5_p5"]].head(10))
    
    print("\n📁 Saving results...")
    out_path = DATA_PROCESSED / "events_with_car.csv"
    
    # ✅ Keep all columns EXCEPT source_url and notes
    cols_to_drop = ["source_url", "notes"]
    cols_to_save = [col for col in events.columns if col not in cols_to_drop]
    events[cols_to_save].to_csv(out_path, sep=";", index=False)
    
    print(f"✅ Saved: {out_path}")
    print(f"   Columns saved: {cols_to_save}")
    
    # Also save prices with AR
    prices_out = DATA_PROCESSED / "prices_with_ar.csv"
    prices.to_csv(prices_out, index=False)
    print(f"✅ Saved: {prices_out}")


if __name__ == "__main__":
    main()