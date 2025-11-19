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