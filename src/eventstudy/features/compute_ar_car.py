# compute_ar_car.py

"""
Compute Abnormal Returns (AR) 
"""

import pandas as pd
import numpy as np
from pathlib import Path

def compute_ar_car():
    """Compute Abnormal Returns (AR)"""
    
    BASE_DIR = Path(__file__).resolve().parents[3]
    DATA_PROCESSED = BASE_DIR / "data" / "processed"

    print(f"Project root: {BASE_DIR}")
    print(f"Data processed: {DATA_PROCESSED}")

    # Load prices with returns
    prices = pd.read_csv(DATA_PROCESSED / "prices_with_returns.csv")
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper()

    print(f"\n✅ Loaded {len(prices)} price records")
    print(prices.head())

    # Estimate alpha & beta for each ticker
    def estimate_alpha_beta(ticker):
        df = prices[prices["ticker"] == ticker].dropna(subset=["return", "market_return"])
        if len(df) < 2:
            return None, None
        
        X = df["market_return"].values
        y = df["return"].values
        X_with_const = np.column_stack([np.ones(len(X)), X])
        params = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        return params[0], params[1]

    # Compute expected returns
    ab_table = {}
    for ticker in prices["ticker"].unique():
        alpha, beta = estimate_alpha_beta(ticker)
        ab_table[ticker] = {"alpha": alpha, "beta": beta}
        print(f"   {ticker}: alpha={alpha:.6f}, beta={beta:.6f}")

    prices["expected_return"] = prices.apply(
        lambda r: ab_table.get(r["ticker"], {}).get("alpha", 0) + 
                  ab_table.get(r["ticker"], {}).get("beta", 0) * r["market_return"]
        if pd.notnull(r["market_return"]) else None,
        axis=1
    )

    prices["AR"] = prices["return"] - prices["expected_return"]

    print(f"\n✅ Computed Abnormal Returns (AR)")
    print(prices.head(10))

    # Save results
    out_path = DATA_PROCESSED / "prices_with_ar.csv"
    prices.to_csv(out_path, index=False)
    print(f"\n✅ Saved to: {out_path}")
    print(f"   Shape: {prices.shape}")
    print(f"   Columns: {prices.columns.tolist()}")
    
    return prices


if __name__ == "__main__":
    compute_ar_car()
