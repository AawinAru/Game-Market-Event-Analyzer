"""
Clean video game sales data for analysis. 
Keep only games from 2013 onwards and specific publishers.
                                    vgsales.csv
"""

import pandas as pd
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent.parent
raw_file = project_root / "data" / "raw" / "vgsales.csv"
output_file = project_root / "data" / "processed" / "vgsales_cleaned.csv"

# Load the data
print("Loading vgsales.csv...")
df = pd.read_csv(raw_file)

print(f"\n{'='*60}")
print("ORIGINAL DATA")
print(f"{'='*60}")
print(f"Total games: {len(df)}")
print(f"Year range: {df['Year'].min()} to {df['Year'].max()}")
print(f"Columns: {df.columns.tolist()}")

# STEP 1: Filter by year (2013 onwards)
print(f"\n{'='*60}")
print("STEP 1: Filtering by Year >= 2013")
print(f"{'='*60}")
df_filtered = df[df['Year'] >= 2013].copy()
print(f"Games after 2013 filter: {len(df_filtered)}")

# STEP 2: Filter by publishers
print(f"\n{'='*60}")
print("STEP 2: Filtering by Publishers")
print(f"{'='*60}")
publishers = ['Take-Two Interactive', 'Ubisoft', 'Activision', 'Nintendo', 'Electronic Arts']
print(f"Keeping only: {publishers}")

df_filtered = df_filtered[df_filtered['Publisher'].isin(publishers)].copy()
print(f"Games from target publishers: {len(df_filtered)}")

# STEP 3: Keep only relevant columns
print(f"\n{'='*60}")
print("STEP 3: Selecting Columns")
print(f"{'='*60}")
print("Keeping: Name, Year, Publisher, Global_Sales")

columns_to_keep = ['Name', 'Year', 'Publisher', 'Global_Sales']
df_cleaned = df_filtered[columns_to_keep].copy()

# STEP 4: Remove any rows with missing data
print(f"\n{'='*60}")
print("STEP 4: Removing Missing Values")
print(f"{'='*60}")
print(f"Missing values before:\n{df_cleaned.isnull().sum()}")

df_cleaned = df_cleaned.dropna()

print(f"\nMissing values after: {df_cleaned.isnull().sum().sum()}")
print(f"Games remaining: {len(df_cleaned)}")

# STEP 5: Summary statistics
print(f"\n{'='*60}")
print("CLEANED DATA SUMMARY")
print(f"{'='*60}")
print(f"\nTotal games: {len(df_cleaned)}")
print(f"Year range: {df_cleaned['Year'].min()} to {df_cleaned['Year'].max()}")
print(f"\nGames by Publisher:")
print(df_cleaned['Publisher'].value_counts())
print(f"\nGlobal Sales Statistics:")
print(df_cleaned['Global_Sales'].describe())

# STEP 6: Save cleaned data
print(f"\n{'='*60}")
print("SAVING CLEANED DATA")
print(f"{'='*60}")
output_file.parent.mkdir(parents=True, exist_ok=True)
df_cleaned.to_csv(output_file, index=False)
print(f"✅ Saved to: {output_file}")

# Show sample
print(f"\n{'='*60}")
print("SAMPLE OF CLEANED DATA (first 10 rows)")
print(f"{'='*60}")
print(df_cleaned.head(10))


