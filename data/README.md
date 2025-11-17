# Data Cleaning Documentation

## Datasets Used

### 1. vgsales.csv ✅ USED
- **Source:** Kaggle - Video Game Sales Data
- **Games:** 16,598 total
- **After filtering (2013+, target publishers):** XXX games
- **Status:** ✅ Used in analysis
- **Columns kept:** Name, Year, Publisher, Global_Sales

### 2. game_statistics_feb_2023.csv ❌ NOT USED
- **Source:** VGChartz Game Statistics
- **Games:** [Total number] total
- **After filtering (2013+, target publishers):** 0 games
- **Status:** ❌ Excluded from analysis
- **Reason:** ALL GAMES RELEASED BEFORE 2013

#### Why Not Used?
The game_statistics_feb_2023.csv dataset contained video game sales and statistics, but **all games in the dataset were released before 2013**. 

Since this project analyzes the impact of game events on stock prices:
- Stock price data available: **2013-2024**
- Game statistics data available: **Before 2013**

**There was no temporal overlap between the two datasets**, making the game_statistics data unusable for the event study methodology.

#### Decision Made
To ensure data consistency and temporal alignment, only **vgsales.csv** (which contains games from 2013 onwards) was used for the final analysis.

---

## Cleaned Data Files

| File | Rows | Columns | Purpose |
|------|------|---------|---------|
| `vgsales_cleaned.csv` | XXX | Name, Year, Publisher, Global_Sales | Game releases and sales data |
| `stock_prices_clean.csv` | XXX | TTWO, EA, ATVI, UBI, NTDOY, ^GSPC | Stock price data for event analysis |
| `stock_returns_clean.csv` | XXX | TTWO, EA, ATVI, UBI, NTDOY, ^GSPC | Daily returns for abnormal return calculation |