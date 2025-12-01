#compute_returns.py

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

#merge_event_returns.py

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
    # ✅ Use utf-8-sig to remove BOM
    events = pd.read_csv(DATA_RAW / "events.csv", sep=";", encoding="utf-8-sig")
    events.columns = events.columns.str.strip()  # Remove extra spaces
    
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
    prices = pd.read_csv(DATA_PROCESSED / "prices_with_returns.csv")
    
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

        # ✅ Ensure dates are datetime before merge
        e["event_date"] = pd.to_datetime(e["event_date"])
        p["trading_date"] = pd.to_datetime(p["trading_date"])

        tmp = pd.merge_asof(
            e.sort_values("event_date"),
            p[["trading_date", "adj_close", "return", "market_return"]].sort_values("trading_date"),
            left_on="event_date",
            right_on="trading_date",
            direction="backward"
        )

        merged_list.append(tmp)

    merged = pd.concat(merged_list, ignore_index=True)
    
    print(f"\n✅ Merged {len(merged)} rows")
    print("Columns:", merged.columns.tolist())
    
    # ✅ Keep only needed columns (exclude source_url and notes)
    cols_to_keep = [
        "event_id", "event_date", "trading_date", "ticker", "publisher", "studio",
        "is_rockstar", "game", "franchise", "event_type", "sentiment", 
        "impact_expectation_manual", "adj_close", "return", "market_return"
    ]
    merged = merged[[col for col in cols_to_keep if col in merged.columns]]

    print(f"\nFinal columns: {merged.columns.tolist()}")
    print(merged.head(10))

    # ✅ Save with semicolon separator
    out_path = DATA_PROCESSED / "events_with_returns.csv"
    merged.to_csv(out_path, sep=";", index=False)
    print(f"\n✅ Saved: {out_path}")
    
    return merged


# === MAIN FUNCTION =========================================================== #

def main():
    print("Loading events...")
    events = load_events()

    print("\nLoading prices...")
    prices = load_prices()

    print("\nMerging...")
    merged = merge_events_with_prices(events, prices)


if __name__ == "__main__":
    main()

#compute_ar_car.py
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

#car_into_labels.py
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

print("BASE_DIR:", BASE_DIR)
print("DATA_PROCESSED exists:", DATA_PROCESSED.exists())


# === LOAD DATA ============================================================= #

def load_data():
    # ✅ Load with correct separator
    events = pd.read_csv(DATA_PROCESSED / "events_with_car.csv", sep=";")
    events["trading_date"] = pd.to_datetime(events["trading_date"])
    events["event_date"] = pd.to_datetime(events["event_date"])
    
    prices = pd.read_csv(DATA_PROCESSED / "prices_with_ar.csv")
    prices["date"] = pd.to_datetime(prices["date"])
    
    return events, prices


# === COMPUTE CAR WINDOWS ================================================== #

def compute_car(prices, ticker, trading_date, start, end):
    """Compute CAR over a custom window [start, end] days from trading_date"""
    mask = (
        (prices["ticker"] == ticker) &
        (prices["date"] >= trading_date + pd.Timedelta(days=start)) &
        (prices["date"] <= trading_date + pd.Timedelta(days=end))
    )
    return prices.loc[mask, "AR"].sum()


def add_car_windows(events, prices):
    """Add multiple CAR window columns"""
    events["CAR_0_1"] = events.apply(
        lambda r: compute_car(prices, r["ticker"], r["trading_date"], 0, 1),
        axis=1
    )
    
    events["CAR_0_3"] = events.apply(
        lambda r: compute_car(prices, r["ticker"], r["trading_date"], 0, 3),
        axis=1
    )
    
    events["CAR_0_5"] = events.apply(
        lambda r: compute_car(prices, r["ticker"], r["trading_date"], 0, 5),
        axis=1
    )
    
    return events


# === CREATE IMPACT LABELS ================================================= #

def label_impact(car):
    """Define ML labels based on CAR magnitude"""
    if abs(car) > 0.03:
        return "High"
    elif abs(car) > 0.01:
        return "Medium"
    else:
        return "Low"


def add_impact_labels(events):
    """Add impact_label column based on CAR_m1_p1"""
    events["impact_label"] = events["CAR_m1_p1"].apply(label_impact)
    return events


# === MAIN ================================================================== #

def main():
    print("📥 Loading data...")
    events, prices = load_data()
    print(f"✅ Loaded {len(events)} events")
    print(f"✅ Loaded {len(prices)} price rows")
    
    print("\n📊 Computing CAR windows...")
    events = add_car_windows(events, prices)
    print("✅ Added CAR_0_1, CAR_0_3, CAR_0_5")
    
    print("\n🏷️  Adding impact labels...")
    events = add_impact_labels(events)
    print("Impact label distribution:")
    print(events["impact_label"].value_counts())
    
    print("\n📁 Saving results...")
    # ✅ Keep all columns EXCEPT source_url and notes
    cols_to_drop = ["source_url", "notes"]
    cols_to_save = [col for col in events.columns if col not in cols_to_drop]
    
    out_path = DATA_PROCESSED / "events_labeled.csv"
    events[cols_to_save].to_csv(out_path, sep=";", index=False)
    print(f"✅ Saved: {out_path}")
    print(f"   Columns: {cols_to_save}")


if __name__ == "__main__":
    main()

#build_ml_dataset.py
import pandas as pd
from pathlib import Path
import glob


BASE_DIR = Path(__file__).resolve().parents[3]
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

print("BASE_DIR:", BASE_DIR)
print("DATA_RAW exists:", DATA_RAW.exists())
print("DATA_PROCESSED exists:", DATA_PROCESSED.exists())


def simplify_event_type(event_type: str) -> str:
    """
    Map detailed event types to simplified categories.
    
    Categories:
    - release: Game releases, launches
    - marketing: Trailers, reveals, gameplay, non-financial announcements
    - negative: Delays, leaks, controversies, lawsuits, failures, warnings
    - corporate: Earnings, corporate events, major financial announcements
    - other: Everything else
    """
    et = str(event_type).strip().lower()

    # RELEASE
    if "release" in et or "launch" in et:
        return "release"

    # MARKETING (trailers, gameplay, screenshots, reveal, non-financial announcements)
    if (
        "trailer" in et
        or "reveal" in et
        or "screenshot" in et
        or "gameplay" in et
        or ("announcement" in et and "earn" not in et and "corporate" not in et)
    ):
        return "marketing"

    # NEGATIVE EVENTS
    if (
        "delay" in et
        or "leak" in et
        or "controvers" in et
        or "lawsuit" in et
        or "warning" in et
        or "failure" in et
    ):
        return "negative"

    # CORPORATE
    if "earn" in et or "corporate" in et or "major" in et:
        return "corporate"

    return "other"


def build_ml_dataset() -> None:
    # ---------- LOAD EVENTS ----------
    in_path = DATA_PROCESSED / "events_labeled.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path, sep=";")
    print("Loaded events_labeled.csv with shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # ---------- BASIC CLEANING ----------

    # sentiment → lowercase + binary negative flag
    if "sentiment" in df.columns:
        df["sentiment"] = (
            df["sentiment"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace("nan", "")
        )
        df["sentiment_negative"] = (df["sentiment"] == "negative").astype(int)
    else:
        df["sentiment_negative"] = 0

    # event_type → lowercase + simplification
    if "event_type" in df.columns:
        df["event_type"] = (
            df["event_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace("nan", "")
        )
        print("\n📊 Event type mapping:")
        print("Before simplification:")
        print(df["event_type"].value_counts())

        df["event_type"] = df["event_type"].apply(simplify_event_type)

        print("\nAfter simplification:")
        print(df["event_type"].value_counts())

    # publisher / franchise → lowercase mainly for GTA flag
    for col in ["publisher", "franchise"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace("nan", "")
            )

    # franchise_gta = 1 if franchise == 'gta'
    if "franchise" in df.columns:
        df["franchise_gta"] = (df["franchise"] == "gta").astype(int)
    else:
        df["franchise_gta"] = 0

    # ensure event dates are datetime (needed for VIX merge)
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"])
    if "trading_date" in df.columns:
        df["trading_date"] = pd.to_datetime(df["trading_date"])

    # ---------- ADD VIX (MARKET VOLATILITY) ----------
    # Find VIX file automatically
    vix_files = list((DATA_RAW).glob("VIX_*.csv"))

    if not vix_files:
        raise FileNotFoundError(f"No VIX file found in {DATA_RAW}")

    vix_path = vix_files[0]  # Use the first VIX file found
    print(f"📈 Loading VIX from: {vix_path}")

    vix = pd.read_csv(vix_path)
    print(f"   VIX columns (raw): {vix.columns.tolist()}")

    # ✅ Standardize column names to lowercase
    vix.columns = [c.lower().strip() for c in vix.columns]
    print(f"   VIX columns (normalized): {vix.columns.tolist()}")

    # Parse date column
    if "date" in vix.columns:
        vix["date"] = pd.to_datetime(vix["date"])
    else:
        raise KeyError(f"VIX file must have a 'Date' column. Found: {vix.columns.tolist()}")

    # ✅ Find VIX value column (flexible detection)
    if "^vix" in vix.columns:
        level_col = "^vix"
    elif "adj close" in vix.columns:
        level_col = "adj close"
    elif "close" in vix.columns:
        level_col = "close"
    elif "vix" in vix.columns:
        level_col = "vix"
    else:
        raise KeyError(
            f"VIX file must have VIX value column (e.g., '^VIX', 'Close', 'Adj Close'). "
            f"Found columns: {vix.columns.tolist()}"
        )

    print(f"   Using column '{level_col}' as VIX level")

    vix = vix.rename(columns={level_col: "vix_level"})
    vix = vix[["date", "vix_level"]].sort_values("date")

    # 30-day rolling mean of VIX
    vix["vix_30d_mean"] = vix["vix_level"].rolling(30, min_periods=1).mean()

    # VIX regime: low / medium / high
    vix["vix_regime"] = pd.qcut(
        vix["vix_level"],
        q=3,
        labels=["low", "medium", "high"]
    )

    print("\n📈 VIX sample:")
    print(vix.head())

    # merge VIX with events on trading_date
    if "trading_date" not in df.columns:
        raise KeyError("Column 'trading_date' is required in events_labeled.csv")

    df = df.merge(
        vix,
        left_on="trading_date",
        right_on="date",
        how="left"
    ).drop(columns=["date"])

    # fill missing vix values (weekends / holidays / early years)
    df["vix_level"] = df["vix_level"].ffill().bfill()
    df["vix_30d_mean"] = df["vix_30d_mean"].ffill().bfill()
    df["vix_regime"] = df["vix_regime"].ffill().bfill()

    print("\n🎯 After VIX merge (first rows):")
    print(df[["event_date", "trading_date", "vix_level", "vix_30d_mean", "vix_regime"]].head())

    # ---------- TARGETS: MULTI-CLASS + BINARY ----------

    target_col = "impact_label"
    if target_col not in df.columns:
        raise KeyError(
            f"Column '{target_col}' not found in {in_path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # 3-class target: low / medium / high
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    label_map = {"low": 0, "medium": 1, "high": 2}
    df["impact_label_num"] = df[target_col].map(label_map)

    if df["impact_label_num"].isna().any():
        bad_vals = df.loc[df["impact_label_num"].isna(), target_col].unique()
        raise ValueError(
            f"Some impact_label values are not in {list(label_map.keys())}: {bad_vals}"
        )

    # binary target for ML: impact_high = 1 if |CAR_m1_p1| >= 3% else 0
    if "CAR_m1_p1" not in df.columns:
        raise KeyError(
            "Column 'CAR_m1_p1' not found in events_labeled.csv. "
            "You must compute CAR_m1_p1 before building the ML dataset."
        )

    df["impact_high"] = (df["CAR_m1_p1"].abs() >= 0.03).astype(int)

    print("\n📌 Target distributions:")
    print("impact_label_num:")
    print(df["impact_label_num"].value_counts().sort_index())
    print("\nimpact_high:")
    print(df["impact_high"].value_counts())

    # ---------- FEATURE SELECTION ----------

    base_features = [
        "is_rockstar",
        "sentiment_negative",
        "market_return",
        "AR_event",
        "franchise_gta",
        "vix_level",
        "vix_30d_mean",
        "event_type",   # categorical
        "vix_regime",   # categorical
    ]

    present_features = [col for col in base_features if col in df.columns]
    print("\n📋 Present feature columns:", present_features)

    if not present_features:
        raise RuntimeError(
            "No ML features found. Check that events_labeled.csv has the expected columns.\n"
            f"Available columns are: {list(df.columns)}"
        )

    # is_rockstar → clean to int 0/1
    if "is_rockstar" in present_features:
        df["is_rockstar"] = (
            pd.to_numeric(
                df["is_rockstar"].astype(str).str.strip(),
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
        )
        print("is_rockstar unique values after cleaning:", df["is_rockstar"].unique())

    # build ML dataframe: features + both targets
    df_ml = df[present_features + ["impact_label", "impact_label_num", "impact_high"]].copy()

    # ---------- ONE-HOT ENCODE CATEGORICALS ----------

    cat_cols = [c for c in ["event_type", "vix_regime"] if c in present_features]
    print("Categorical columns to encode:", cat_cols)

    df_ml_encoded = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)

    before = len(df_ml_encoded)
    df_ml_encoded = df_ml_encoded.dropna(subset=["impact_label_num", "impact_high"])
    after = len(df_ml_encoded)
    print(f"Dropped {before - after} rows with missing labels.")

    # final feature list
    feature_cols = [
        c
        for c in df_ml_encoded.columns
        if c not in ["impact_label", "impact_label_num", "impact_high"]
    ]

    df_ml_encoded = df_ml_encoded.replace({True: 1, False: 0})
    
    print("\n✅ ML DATASET SUMMARY")
    print("=" * 70)
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Total rows: {len(df_ml_encoded)}")

    print("\nLabel distribution (impact_label_num):")
    print(df_ml_encoded["impact_label_num"].value_counts().sort_index())

    print("\nBinary target distribution (impact_high):")
    print(df_ml_encoded["impact_high"].value_counts())

    print("\nFeature names:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    # ---------- SAVE ----------
    out_path = DATA_PROCESSED / "ml_dataset.csv"
    df_ml_encoded.to_csv(out_path, sep=";", index=False)
    print(f"\n✅ Saved ML dataset to: {out_path}")


if __name__ == "__main__":
    build_ml_dataset()

#trends.py
"""
Build regression dataset with GTA-only Google Trends features.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import time
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

print(f"BASE_DIR:       {BASE_DIR}")
print(f"DATA_RAW:       {DATA_RAW}")
print(f"DATA_PROCESSED: {DATA_PROCESSED}\n")


def download_gta_trends_or_use_cache(start_date="2010-01-01", end_date="2025-12-31", geo="US"):
    """Download or use cached GTA trends."""
    cache_path = DATA_RAW / "gta_trends.csv"
    
    if cache_path.exists():
        print(f"📂 Using CACHED GTA trends from: {cache_path}")
        df = pd.read_csv(cache_path)
        print(f"✅ Loaded {len(df)} rows from cache\n")
        return df
    
    print("🔧 Creating MOCK GTA trends data for testing...\n")
    
    dates = pd.date_range(start="2010-01-01", end="2025-12-31", freq="W")
    n = len(dates)
    
    mock_data = pd.DataFrame({
        "date": dates,
        "trend_gta": np.random.randint(20, 100, n),
        "trend_gta6": np.random.randint(10, 80, n),
        "trend_gtavi": np.random.randint(5, 60, n),
        "trend_rockstar": np.random.randint(15, 70, n),
    })
    
    for col in ["trend_gta", "trend_gta6", "trend_gtavi", "trend_rockstar"]:
        mock_data[col] = mock_data[col].rolling(4, center=True).mean().fillna(mock_data[col])
    
    mock_data["trend_gta_mean"] = mock_data[["trend_gta", "trend_gta6", "trend_gtavi"]].mean(axis=1)
    
    print("✅ Mock GTA trends sample:")
    print(mock_data.head(10), "\n")
    
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    mock_data.to_csv(cache_path, index=False)
    print(f"✅ Saved mock data to: {cache_path}\n")
    
    return mock_data


def load_events_labeled():
    """Load events_labeled.csv from DATA_PROCESSED."""
    path = DATA_PROCESSED / "events_labeled.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    print(f"📥 Loading events_labeled from: {path}")
    df = pd.read_csv(path, sep=";")

    print(f"📋 Available columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    print()

    # Flexible date column detection
    date_col = None
    for col_name in ["event_date", "date", "event_date_str", "trading_date"]:
        if col_name in df.columns:
            date_col = col_name
            break
    
    if date_col is None:
        raise KeyError(
            f"No date column found! Expected one of: "
            f"['event_date', 'date', 'event_date_str', 'trading_date']. "
            f"Found columns: {df.columns.tolist()}"
        )
    
    print(f"✅ Parsing dates from column: '{date_col}'")
    df["event_date"] = pd.to_datetime(df[date_col], errors="coerce")
    
    if "trading_date" in df.columns:
        df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    else:
        print("⚠️  Creating trading_date from event_date...")
        df["trading_date"] = df["event_date"]

    print(f"✅ Loaded events_labeled: {df.shape}")
    print(f"   Date range: {df['event_date'].min()} to {df['event_date'].max()}\n")

    return df


def expand_trends_to_daily(trends):
    """Convert weekly GTA trends to daily series via forward-fill."""
    print("📆 Expanding weekly GTA trends to DAILY (forward-fill)...")

    t = trends.copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t.set_index("date")

    t_daily = t.resample("D").ffill().reset_index()

    print("✅ GTA daily trends sample:")
    print(t_daily.head(), "\n")

    return t_daily


def normalize_trend_columns(df):
    """Add z-scored versions of trend columns."""
    trend_cols = ["trend_gta", "trend_gta6", "trend_gtavi", "trend_rockstar", "trend_gta_mean"]

    for col in trend_cols:
        if col not in df.columns:
            continue
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            df[col + "_z"] = 0.0
        else:
            df[col + "_z"] = (df[col] - mean) / std

    print("✅ Added normalized trend columns (_z)\n")
    return df


def build_regression_dataset_with_gta_trends():
    """Main function to build regression dataset with GTA trends."""
    
    trends_weekly = download_gta_trends_or_use_cache()
    trends_daily = expand_trends_to_daily(trends_weekly)
    events = load_events_labeled()

    print("🔗 Merging GTA trends with events on trading_date...")
    merged = events.merge(
        trends_daily,
        left_on="trading_date",
        right_on="date",
        how="left",
    )

    merged = merged.drop(columns=["date"])

    print("✅ Sample of merged regression dataset (with raw trends):")
    print(
        merged[
            [
                "event_id",
                "trading_date",
                "ticker",
                "AR_event",
                "CAR_0_1",
                "trend_gta",
                "trend_gta6",
                "trend_gtavi",
                "trend_rockstar",
                "trend_gta_mean",
            ]
        ].head(),
        "\n",
    )

        # === FIX: Only TTWO events should keep GTA/GTAVI/GTA6 trends ===
    trend_cols = [
        c for c in merged.columns
        if c.startswith("trend_gta")
        or c.startswith("trend_gtavi")
        or c.startswith("trend_rockstar")
    ]

    mask_non_ttwo = merged["ticker"] != "TTWO"
    merged.loc[mask_non_ttwo, trend_cols] = None

    print(f"✔ GTA trends restricted to TTWO only — Cleared {mask_non_ttwo.sum()} rows.\n")
    
    merged = normalize_trend_columns(merged)

    out_path = DATA_PROCESSED / "regression_dataset_with_gta_trends.csv"
    merged.to_csv(out_path, index=False, sep=";")
    print(f"✅ Saved regression dataset with GTA trends to: {out_path}\n")

    return merged


def inspect_events_labeled_file():
    """Inspect the events_labeled.csv to see its actual structure."""
    path = DATA_PROCESSED / "events_labeled.csv"
    
    if not path.exists():
        print(f"❌ File not found: {path}")
        return
    
    df = pd.read_csv(path, sep=";")
    
    print("="*70)
    print("📊 EVENTS_LABELED.CSV INSPECTION")
    print("="*70)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col:30s} | {df[col].dtype}")
    
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    print(df.isnull().sum())


# ---- MAIN ----
def main():
    print("🚀 Building regression dataset with GTA-only Google Trends...\n")
    
    # ✅ First, inspect the file structure
    print("🔍 Inspecting events_labeled.csv...\n")
    inspect_events_labeled_file()
    
    print("\n" + "="*70 + "\n")
    
    # Then try to load
    try:
        df_reg = build_regression_dataset_with_gta_trends()
        print("📊 Final regression dataset shape:", df_reg.shape)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Please check the column names in events_labeled.csv")


if __name__ == "__main__":
    main()