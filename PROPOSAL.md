# Game Market Event Analyzer
## Project Category: Financial Data Analysis

### Problem Statement / Motivation
The video game industry represents a significant market segment where major game releases can substantially impact stock prices. The upcoming GTA VI release presents a unique opportunity to study these market dynamics. This project aims to answer: How do major video-game events influence the short-term stock-market performance of publishers, and can past patterns help estimate the expected impact of GTA VI's release?

### Planned Approach & Technologies
1. **Data Collection & Processing**
   - Stock price data via `yfinance` for publishers (TTWO, EA, UBSFY, ATVI) and market index
   - Video game sales and event data from Kaggle datasets
   - Manual compilation of major events (launches, trailers, delays)
   - Market volatility data (VIX) for control variables

2. **Analysis Framework**
   - Event study methodology to calculate abnormal returns
   - OLS regression analysis: `CARi = α + β1Salesi + β2EventTypei + β3Franchisei + β4Publisheri + β5VIXi + εi`
   - Python packages: pandas, statsmodels, scipy

3. **Implementation**
   - Modular Python package with CLI interface
   - Data visualization using matplotlib/plotly
   - Validation using recent GTA VI-related events

### Expected Challenges & Solutions
1. **Data Quality & Collection**
   - Challenge: Incomplete event data and dates
   - Solution: Multiple data sources and manual verification

2. **Statistical Significance**
   - Challenge: Limited sample size for major game releases
   - Solution: Include different types of events and robust statistical methods

3. **Model Validation**
   - Challenge: Unique nature of GTA VI release
   - Solution: Validate on recent events (e.g., May 2025 delay announcement)

### Success Criteria
1. **Technical Metrics**
   - Functioning event study pipeline
   - Model explains >60% of price variation
   - Successful validation on recent GTA VI events

2. **Deliverables**
   - Reproducible Python package
   - Comprehensive documentation
   - Visual analysis of results
   - Statistical validation of findings

### Stretch Goals
1. Integration of social media sentiment analysis
2. Interactive dashboard for real-time analysis
3. Extension to other gaming markets
4. Options market data integration

Word count: 347



