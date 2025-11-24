import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

print("BASE_DIR:", BASE_DIR)
print("DATA_RAW exists:", DATA_RAW.exists())
print("DATA_PROCESSED exists:", DATA_PROCESSED.exists())

def build_ml_dataset() -> None:
    """
    Build a clean ML-ready dataset from events_labeled.csv.

    Steps:
    - Locate project root from this file's path
    - Load data/processed/events_labeled.csv
    - Normalize text fields (e.g. sentiment)
    - Select useful features
    - One-hot encode categorical variables
    - Map impact_label -> numeric labels
    - Save data/processed/ml_dataset.csv
    """

    # ---- Paths ----
    in_path = DATA_PROCESSED / "events_labeled.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # ✅ ADD sep=";" to load correctly
    df = pd.read_csv(in_path, sep=";")
    print("Loaded events_labeled.csv with shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # ---- Basic normalization ----
    # ONLY lowercase sentiment
    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    
    # Keep other columns as-is (just strip whitespace)
    for col in ["event_type", "franchise", "publisher", "studio"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ---- Target column ----
    target_col = "impact_label"
    if target_col not in df.columns:
        raise KeyError(
            f"Column '{target_col}' not found in {in_path}. "
            "Available columns: " + str(df.columns.tolist())
        )

    # Normalize impact_label case just in case
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()

    # Map labels to numbers: low=0, medium=1, high=2
    label_map = {"low": 0, "medium": 1, "high": 2}
    df["impact_label_num"] = df[target_col].map(label_map)

    if df["impact_label_num"].isna().any():
        bad_vals = df.loc[df["impact_label_num"].isna(), target_col].unique()
        raise ValueError(
            f"Some impact_label values are not in {list(label_map.keys())}: {bad_vals}"
        )

    # ---- Candidate features ----
    base_features = [
        "publisher",
        "studio",
        "is_rockstar",
        "event_type",
        "franchise",
        "sentiment",
        # market context / returns
        "market_return",
        "AR_event",
        # CAR windows
        "CAR_0_1",
        "CAR_m1_p1",
        "CAR_0_3",
        "CAR_0_5",
        "CAR_m5_p5",
    ]

    present_features = [col for col in base_features if col in df.columns]
    print("Present feature columns:", present_features)

    if not present_features:
        raise RuntimeError(
            "No ML features found. Check that events_labeled.csv has the expected columns.\n"
            f"Available columns are: {list(df.columns)}"
        )

    # ---- Handle is_rockstar as int (0/1) if present ----
    if "is_rockstar" in present_features:
        # ✅ Strip whitespace, replace empty strings with 0, then convert to int
        df["is_rockstar"] = (
            df["is_rockstar"]
            .astype(str)
            .str.strip()  # Remove leading/trailing spaces
            .replace("", "0")  # Replace empty strings with 0
            .replace("nan", "0")  # Replace 'nan' strings with 0
            .astype(int)
        )
        print(f"is_rockstar unique values after cleaning: {df['is_rockstar'].unique()}")

    # Build ML dataframe (features + targets)
    df_ml = df[present_features + [target_col, "impact_label_num"]].copy()

    # ---- One-hot encode categorical columns ----
    cat_cols = [
        col
        for col in ["publisher", "studio", "event_type", "franchise", "sentiment"]
        if col in present_features
    ]

    print("Categorical columns to encode:", cat_cols)

    df_ml_encoded = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)

    # Drop rows with missing label
    before = len(df_ml_encoded)
    df_ml_encoded = df_ml_encoded.dropna(subset=["impact_label_num"])
    after = len(df_ml_encoded)
    print(f"Dropped {before - after} rows with missing labels.")

    # ---- Final feature/target structure ----
    feature_cols = [
        c for c in df_ml_encoded.columns if c not in [target_col, "impact_label_num"]
    ]

    print("ML dataset – feature columns:", len(feature_cols))
    print("ML dataset – total rows:", len(df_ml_encoded))
    print("Label distribution (impact_label_num):")
    print(df_ml_encoded["impact_label_num"].value_counts().sort_index())

    # ---- Save final ML dataset ----
    out_path = DATA_PROCESSED / "ml_dataset.csv"
    df_ml_encoded.to_csv(out_path, sep=";", index=False)  # ✅ Also save with sep=";"
    print(f"✅ Saved ML dataset to: {out_path}")


if __name__ == "__main__":
    build_ml_dataset()