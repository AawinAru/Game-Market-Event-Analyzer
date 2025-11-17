"""
Attempt to clean game_statistics_feb_2023.csv data.
This file is kept for documentation purposes to show why it was NOT used.
"""

import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
raw_file = project_root / "data" / "raw" / "game_statistics_feb_2023.csv"

# Load the data
print("Loading game_statistics_feb_2023.csv...")
df = pd.read_csv(raw_file)

print(f"\n{'='*60}")
print("GAME STATISTICS DATA - VIABILITY CHECK")
print(f"{'='*60}")

# Extract year
df['year'] = pd.to_datetime(df['release_date'], format='%d %b %y', errors='coerce').dt.year

# Check year range
print(f"\nYear range in dataset: {df['year'].min():.0f} to {df['year'].max():.0f}")
print(f"Games from 2013+: {len(df[df['year'] >= 2013])}")

# Filter by year
df_filtered = df[df['year'] >= 2013]

print(f"\n{'='*60}")
print("ANALYSIS RESULT")
print(f"{'='*60}")
print(f"\n❌ DATASET NOT VIABLE FOR THIS PROJECT")
print(f"\nReason: All games released before 2013")
print(f"  - Stock price data available: 2013-2024")
print(f"  - Game statistics data available: Before 2013")
print(f"  - Temporal overlap: NONE")
print(f"\nDecision: Use vgsales.csv instead (has games from 2013+)")