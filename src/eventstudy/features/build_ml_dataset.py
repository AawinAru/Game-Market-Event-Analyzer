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