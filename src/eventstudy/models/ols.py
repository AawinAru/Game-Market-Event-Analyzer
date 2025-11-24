import pandas as pd
import statsmodels.api as sm
from pathlib import Path

DATA = Path("../../data/processed/events_with_car.csv")

df = pd.read_csv(DATA, sep=";")

# --------------------------
# 1. Normalize text columns
# --------------------------
text_cols = ["publisher", "studio", "event_type", "sentiment", "franchise"]

for col in text_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace("nan", "")
        )

# --------------------------
# 2. Create GTA binary
# --------------------------
df["franchise_gta"] = (df["franchise"] == "gta").astype(int)

# --------------------------
# 3. Force all numeric columns
# --------------------------
num_cols = [
    "car_m1_p1",
    "ar_event",
    "market_return",
    "is_rockstar",
    "franchise_gta",
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["car_m1_p1"])

# --------------------------
# 4. Build feature matrix
# --------------------------
feature_cols = [
    "franchise_gta",
    "is_rockstar",
    "market_return",
    "ar_event",
    "event_type",
    "publisher",
    "sentiment",
]

X_base = df[feature_cols]

# One-hot encoding AFTER standardization
X = pd.get_dummies(
    X_base,
    columns=["event_type", "publisher", "sentiment"],
    drop_first=True
)

# Convert all remaining to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Dependent variable
y = pd.to_numeric(df["car_m1_p1"], errors="coerce")

# Drop rows with missing values
mask = y.notna() & X.notna().all(axis=1)
X = X.loc[mask]
y = y.loc[mask]

# --------------------------
# 5. Fit OLS
# --------------------------
X = sm.add_constant(X)
ols_model = sm.OLS(y, X).fit()

print(ols_model.summary())