# car_into_labels.py

"""
Cumulative Abnormal Returns (CAR) and Create impact labels from CAR.
"""

import pandas as pd
from pathlib import Path

def car_into_labels():
    """Create impact labels from CAR"""
    
    BASE_DIR = Path(__file__).resolve().parents[3]
    DATA_PROCESSED = BASE_DIR / "data" / "processed"

    print(f"Project root: {BASE_DIR}")
    print(f"Data processed: {DATA_PROCESSED}")

    # Load events and prices
    events = pd.read_csv(DATA_PROCESSED / "events_with_returns.csv", sep=";")
    prices = pd.read_csv(DATA_PROCESSED / "prices_with_ar.csv")
    prices["date"] = pd.to_datetime(prices["date"])

    print(f"\n✅ Loaded {len(events)} events")
    print(events.head())

    # Compute CAR function
    def compute_car(ticker, event_date, start_day, end_day):
        """Compute CAR for a window around event date"""
        if pd.isna(event_date):
            return None
        
        event_date = pd.to_datetime(event_date)
        start = event_date + pd.Timedelta(days=start_day)
        end = event_date + pd.Timedelta(days=end_day)
        
        mask = (
            (prices["ticker"] == ticker) &
            (prices["date"] >= start) &
            (prices["date"] <= end)
        )
        return prices.loc[mask, "AR"].sum()

    events["date"] = pd.to_datetime(events["date"])
    
    events["trading_date"] = events["date"]

    def get_ar_event(row):
        if pd.isna(row["date"]):
            return None
        mask = (
            (prices["ticker"] == row["ticker"]) &
            (prices["date"] == row["date"])
        )
        vals = prices.loc[mask, "AR"]
        if vals.empty:
            return None
        return vals.iloc[0]

    events["AR_event"] = events.apply(get_ar_event, axis=1)

    # Compute CAR windows
    print(f"\n🔨 Computing CAR windows...")
    events["CAR_m1_p1"] = events.apply(
        lambda r: compute_car(r["ticker"], r["date"], -1, 1), axis=1
    )
 
    events["CAR_0_1"] = events.apply(
        lambda r: compute_car(r["ticker"], r["date"], 0, 1), axis=1
    )
    events["CAR_0_3"] = events.apply(
        lambda r: compute_car(r["ticker"], r["date"], 0, 3), axis=1
    )
    events["CAR_0_5"] = events.apply(
        lambda r: compute_car(r["ticker"], r["date"], 0, 5), axis=1
    )

    print(f"✅ CAR windows computed")
    print(
        events[
            [
                "ticker",
                "date",
                "trading_date",
                "AR_event",      
                "CAR_m1_p1",
                "CAR_0_1",       
                "CAR_0_3",
                "CAR_0_5",
            ]
        ].head()
    )

    # Create labels based on CAR
    def label_impact(car):
        if pd.isna(car):
            return None
        if abs(car) > 0.03:
            return "High"
        elif abs(car) > 0.01:
            return "Medium"
        else:
            return "Low"

    events["impact_label"] = events["CAR_m1_p1"].apply(label_impact)
    events["impact_label_num"] = events["impact_label"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    )
    events["impact_high"] = (events["impact_label_num"] == 2).astype(int)

    print(f"\n✅ Impact labels created")
    print(f"   Label distribution:")
    print(events["impact_label"].value_counts())

    # Save results
    out_path = DATA_PROCESSED / "events_labeled.csv"
    events.to_csv(out_path, sep=";", index=False)
    print(f"\n✅ Saved to: {out_path}")
    print(f"   Shape: {events.shape}")
    print(f"   Columns: {events.columns.tolist()}")
    
    return events


if __name__ == "__main__":
    car_into_labels()