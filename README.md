# Game Market Event Analyzer

## Research Question
How do video game events (announcements, delays) impact Take-Two Interactive (TTWO) stock returns? Can OLS regression and machine learning predict abnormal returns?

## Project Structure

```
game-market-event-analyzer/
├── README.md                    # This file
├── PROPOSAL.md                  # Original project proposal
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── main.py                      # Full pipeline execution
│
├── src/eventstudy/
│   ├── data/
│   │   └── data_loader.py              # Load event dataset and build long_prices.csv
│   │
│   ├── features/
│   │   ├── compute_returns.py          # Daily returns calculation
│   │   ├── merge_event_returns.py      # Match events to trading dates
│   │   ├── compute_ar_car_returns.py   # Abnormal & cumulative returns
│   │   ├── car_into_labels.py          # Create impact labels (Low/Med/High)
│   │   ├── build_ml_dataset.py         # ML feature engineering
│   │   └── trends.py                   # Add Google Trends features
│   │
│   └── models/
│       ├── ols.py                      # OLS regression (3 models)
│       ├── train_multiclass.py         # Multiclass classification (4 models)
│       ├── train_binary.py             # Binary classification (4 models)
│       ├── backtest_ols.py             # OLS hold-out validation
│       ├── backtest_ml.py              # ML hold-out validation
│       └── gta6_prediction.py          # GTA VI scenario predictions
│   
│   
│       
│
├── data/
│   ├── raw/
│   │   ├── events.csv                  # Game events dataset
│   │   ├── [Ticker]_2010_2025.csv      # Stocks prices from companies, SP500 and VIX
│   │   ├── gta_trends.csv              # GTA trends from Google
│   │   └── backtest.csv                # Hold-out GTA VI events
│   │
│   └── processed/
│       ├── prices_long.csv           # Daily stock prices
│       ├── prices_with_ar.csv        # With abnormal returns
│       ├── events_with_returns.csv   # Events + AR values
│       ├── events_labeled.csv        # Events + impact labels
│       └── ml_dataset.csv
│
├── results/
│   ├── 02_ols_regression/
│   │   ├── ols_regression_results.txt       # OLS summary stats
│   │   ├── ols_regression_coeffs.csv        # Model coefficients
│   │   ├── gta_scenario_predictions.csv     # 3×3 scenario grid
│   │   ├── ols_models.pkl                   # Saved OLS models
│   │   └── ols_feature_names.pkl            # Feature names
│   │
│   ├── 03_binary_classification/
│   │
│   ├── 04_multiclass_classification/
│   │
│   ├── 05_summary/
│   │   ├── gta6_backtest_ols_model2.csv             # OLS backtest
│   │   ├── gta6_backtest_ml_gradient_boosting.csv   # Multiclass backtest
│   │   ├── gta6_backtest_ml_binary_logistic.csv     # Binary backtest
│   │   └── gta6_scenario_predictions_binary.csv     # Binary scenarios
│   │
│   └── figures/
│       ├── figure5_gta6_scenarios.png       # Scenario heatmaps (OLS + Binary)
│       └── figure6_gta6_backtests.png       # Backtest results (3 panels)
│
└── notebooks/
   └── visualize_gta6.ipynb    # Jupyter analysis & visualization if you want to see graph
 
```

---

### Installation

```bash
# 1. Navigate to project directory
cd capstone_project/game-market-event-analyzer

# 2. Create Conda environment
conda env create -f environment.yml

# 3. Activate environment
conda activate game-market-event-analyzer

### Run Full Pipeline
# 4. Execute all steps (data → features → models → backtest → visualization)
python main.py 
python main.py | tee pipeline_output.log # if you want to see the whole terminal output

```bash
# Or run individual steps
python src/eventstudy/data/data_loader.py                       # Step 1: Fetch prices
python src/eventstudy/features/compute_returns.py               # Step 2: Calculate return
python src/eventstudy/features/merge_event_returns.py           # Step 3: Merge events
python src/eventstudy/features/compute_ar_car_returns.py        # Step 4: Calculate AR
python src/eventstudy/features/car_into_labels.py               # Step 5: Create labels
python src/eventstudy/features/build_ml_dataset                 # Step 6: Build ML dataset
python src/eventstudy/features/trends.py                        # Step 7: Add Google trend
python src/eventstudy/models/ols.py                             # Step 8: Train OLS
python src/eventstudy/models/train_multiclassification.py       # Step 9: Multiclass ML
python src/eventstudy/models/train_binary.py                    # Step 10: Binary ML
python src/eventstudy/models/gta6_prediction.py                 # Step 11: Scenarios
python src/eventstudy/models/backtest_ols.py                    # Step 12: OLS validation
python src/eventstudy/models/backtest_ml.py                     # Step 13: ML validation
python notebooks/visualize_gta6.ipynb                           # Step 14: Generate figure
```
## Results

### Data Summary
- **Events analyzed:** 151 game events (2011-2025)
- **Trading days:** 23,958 (2010-01-04 to 2025-11-14)
- **Tickers:** TTWO (Take-Two), EA, ATVI, NTDOY, UBI, ^GSPC (S&P 500)
- **Impact labels:** High (63), Medium (50), Low (38)

### OLS Regression Models

| Model | Features | R² | LOO-MAE | Notes |
|-------|----------|-----|---------|-------|
| **Model 1** | Market return + Sentiment + Event type | 0.189 | 0.0158 | Baseline |
| **Model 2** | Model 1 + GTA Google Trends | **0.227** | 0.0166 | **Best fit** |
| **Model 3** | Model 2 (TTWO-only) | 0.255 | 0.0260 | Small sample (n=48) |

**Key OLS Coefficients (Model 2):**
- `is_rockstar`: +0.0095 (Rockstar games command premium)
- `market_return`: -0.444 (inverse relationship)
- `sent_negative`: -0.0130 (negative sentiment hurts)
- `trend_rockstar_z`: +0.0063 (Google Trends boost, marginal)

### Binary Classification Models

| Model | Test Accuracy | 5-Fold CV | Notes |
|-------|---------------|-----------|-------|
| Logistic Regression | **68.4%** | 63.6% ± 6.9% | **Best performer** |
| Gradient Boosting | **68.4%** | 58.9% ± 3.3% | High overfitting |
| Random Forest | 60.5% | 60.3% ± 5.6% | Moderate fit |
| Neural Network (MLP) | 55.3% | 53.6% ± 2.2% | Underfitting |

### Multiclass Classification Models

| Model | Test Accuracy | 5-Fold CV | Notes |
|-------|---------------|-----------|-------|
| Gradient Boosting | **50.0%** | 39.0% ± 3.9% | Best accuracy |
| Logistic Regression | 44.7% | 50.4% ± 6.5% | Better CV |
| Random Forest | 44.7% | 47.0% ± 0.7% | Stable |
| Neural Network (MLP) | 34.2% | 35.7% ± 8.7% | Weak performance |

### GTA VI Scenario Analysis (3×3 Grid)

**OLS Model 2 Predictions:**

| Market | Low Hype | Medium Hype | High Hype |
|--------|----------|------------|-----------|
| **Bear** | +0.0182 | +0.0257 | +0.0406 |
| **Base** | +0.0271 | +0.0346 | +0.0495 |
| **Bull** | +0.0360 | +0.0434 | +0.0584 |

**Interpretation:**
- Range: +1.82% to +5.84% abnormal return
- **Best case:** Bull market + High hype (+5.84%)
- **Worst case:** Bear market + Low hype (+1.82%)
- **GTA trend effect:** ~4% swing between low/high hype

### GTA VI Backtest Results (Hold-Out Event)

**Event:** TTWO_2025_GTA6_DELAY1 (May 2, 2025)

| Model | True Value | Prediction | Result |
|-------|-----------|-----------|--------|
| **OLS Model 2** | -0.0810 | -0.0113 | ❌ Underpredicted |
| **Multiclass GB** | High | Medium | ⚠️ Misclassified |
| **Binary LR** | High | High | ✅ Correct |

**Summary:**
- OLS underpredicts magnitude of delays (MAE: 0.0697)
- Binary model correctly identifies high-impact events
- Multiclass struggles with Medium/High boundary

## Feature Engineering

### ML Features (11 total)
1. `is_rockstar` — Binary (Rockstar Games)
2. `sentiment_negative` — Binary
3. `market_return` — Daily S&P 500 return
4. `franchise_gta` — Binary
5. `vix_level` — Volatility index
6. `vix_30d_mean` — 30-day rolling average
7. `event_type_marketing` — Binary
8. `event_type_negative` — Binary
9. `event_type_release` — Binary
10. `vix_regime_medium` — Binary
11. `vix_regime_high` — Binary

### OLS Features (15 total)
- All ML features +
- Sentiment dummies (negative, positive)
- Event type dummies (delay, earnings, leak, major announcement, release, trailer/reveal)
- GTA Google Trends z-scores (trend_gta_z, trend_gta6_z, trend_gtavi_z, trend_rockstar_z)

## Methodology

### Event Study Framework
```
AR_event = Actual Return - Expected Return (Market Model)
Expected Return = α + β × Market Return
CAR = Cumulative Abnormal Return over event window
```

**Event windows analyzed:**
- `-1 to +1`: Immediate (3-day)
- `0 to +3`: Short-term
- `0 to +5`: Medium-term

**Estimation window:** 120 trading days pre-event

### Impact Labels
- **High:** CAR(0,5) > median (> 0.0084)
- **Medium:** CAR(0,5) between 25th-75th percentile
- **Low:** CAR(0,5) < 25th percentile (< -0.0011)

## Data Sources

| Source | Coverage | Records |
|--------|----------|---------|
| Yahoo Finance (yfinance) | 2010-01-04 to 2025-11-14 | 23,958 daily prices |
| Manual event research | 151 game events | 1 event per publisher |
| Google Trends API | Search volume (GTA) | 835 weekly readings |
| S&P 500 (^GSPC) | Market benchmark | Aligned with prices |
| VIX (^VIX) | Market volatility | Daily readings |

## Key Findings

✅ **OLS Model 2 R² = 0.227** — Google Trends + sentiment explain ~23% of AR variance  
✅ **Binary models > Multiclass** — Easier to predict High vs. Not High (68% vs 50%)  
✅ **Logistic Regression wins for classification** — Simple, no overfitting (68.4% test accuracy)  
✅ **GTA franchise premium** — Rockstar games get +0.95% coefficient boost  
✅ **Delays underestimated by OLS** — Model predicts -1.1% but actual is -8.1%  
✅ **Market timing crucial** — Bull market scenarios show +4% boost vs. bear  

## Model Selection & Recommendations

### Best OLS Model: **Model 2** (with GTA Trends)
- **Pros:** Highest R² (0.227), includes trends
- **Cons:** Still explains only 23% of variance
- **Use case:** Baseline predictions, interpretation
### Best Binary Classifier: **Logistic Regression**
- **Pros:** 68.4% accuracy, good generalization, interpretable
- **Cons:** Linear boundary, misses edge cases
- **Use case:** Production predictions for High/Not High impact

### Best Multiclass Classifier: **Gradient Boosting**
- **Pros:** 50% accuracy, captures non-linearity
- **Cons:** 46.5% overfitting gap, needs more data
- **Use case:** Research, but not production-ready

## Limitations & Future Work

⚠️ **Current Limitations:**
- Limited GTA VI data (game still in development)
- Outlier events create high skew in residuals (skew=1.48)
- Small TTWO sample size (n=48 for Model 3)
- No macroeconomic factors (interest rates, sector trends)
- Google Trends weak predictors (t-stat=1.850, p=0.067)

🔮 **Future Extensions:**
- [ ] Expand to other publishers 
- [ ] Add sentiment from financial news headlines
- [ ] GARCH models for volatility clustering
- [ ] Real-time prediction API (FastAPI)
- [ ] Time-series features (LSTM for sequential patterns)
- [ ] Causal inference (propensity matching)


## Output Files

**CSV Results:**
- `results/02_ols_regression/ols_regression_coeffs.csv` — All model coefficients
- `results/03_binary_classification/binary_test_results.csv` — Binary predictions
- `results/04_multiclass_classification/multiclass_test_results.csv` — Multiclass predictions
- `results/05_summary/gta6_backtest_*.csv` — Hold-out validation results

**Visualizations (300 DPI):**
- `results/figures/figure5_gta6_scenarios.png` — OLS + Binary scenario heatmaps
- `results/figures/figure6_gta6_backtests.png` — Backtest results (3 panels)
- `results/0X_*/confusion_matrix.png` — Model confusion matrices

**Models:**
- `results/02_ols_regression/ols_models.pkl` — Fitted OLS models
- `results/02_ols_regression/ols_feature_names.pkl` — Feature metadata

## Dependencies

```
Core: pandas, numpy, scikit-learn, statsmodels, xgboost
Visualization: matplotlib, seaborn
Data: yfinance, pytrends
```

Full list: `requirements.txt`
