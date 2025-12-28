# Capstone Project Proposal  
**Game Market Event Analyzer: Quantifying and Simulating Market Reactions to Major Video Game Releases**

**Student:** Aawin Arulampalam  
**Student ID:** 22408330  
**Course:** Data Science and Advanced Programming - HEC Lausanne

---

## Overview / Motivation

### Research Question  
**How do major video-game events influence the short-term stock-market performance of publishers, and can past patterns help estimate the expected impact of GTA VI’s release?**

This project analyzes how the stock prices of leading video-game publishers — **Take-Two Interactive (TTWO), Electronic Arts (EA), Ubisoft (UBSFY), Activision-Blizzard (ATVI), and Nintendo (NTDOY)** — react to key industry events such as blockbuster launches, trailer announcements, delays, and earnings releases.

The goal is to determine whether product success (sales, popularity, franchise strength) drives significant short-term abnormal returns in the stock market. A special focus is placed on **Rockstar Games titles**, including *GTA V*, *Red Dead Redemption 2*, and the upcoming *GTA VI*.

Using historical evidence, the project aims to **simulate the likely market effect of the GTA VI release**, while also validating the model on recent real-world events such as the **May 2025 GTA VI delay announcement**. This work bridges **data science and financial economics**, applying event-study methodology, regression analysis, and structured data pipelines to real-world financial and gaming datasets.

---

## Planned Approach & Methodology

### 1. Data Collection

- **Stock Prices:**  
  Daily adjusted close prices for TTWO, EA, ATVI, UBI, NTDOY, and the S&P 500, ideally sourced from Yahoo Finance via `yfinance` or Kaggle.

- **Video-Game Sales:**  
  Launch revenue, units sold, publisher, and franchise data from Kaggle datasets such as *“Video Game Sales 2024 Update”* and *“Video Game Sales with Ratings”*.

- **Event Data:**  
  A manually curated list of major events (launches, trailers, delays, earnings) stored in `data/events.csv`.

- **Market Volatility:**  
  CBOE Volatility Index (VIX, `^VIX`) from Yahoo Finance to control for overall market conditions.

- **Optional Attention Data:**  
  Google Trends indices for terms such as *“GTA 6”* and *“Call of Duty”* collected via `pytrends`.

- **External Validation Data:**  
  Analyst forecasts for GTA VI revenues and valuation scenarios from equity research reports (e.g., WUTIS 2025), as well as market-implied expectations or volatility derived from options data.

---

### 2. Analysis

- Compute daily stock returns and estimate expected (“normal”) returns using a market model.
- Calculate **Abnormal Returns (AR)** and **Cumulative Abnormal Returns (CAR)** over event windows such as \([-1, +1]\) and \([0, +5]\).
- Estimate an OLS regression model of the form:

\[
CAR_i = \alpha + \beta_1 Sales_i + \beta_2 EventType_i + \beta_3 Franchise_i + \beta_4 Publisher_i + \beta_5 VIX_i + \varepsilon_i
\]

#### Validation Strategy

- **Validation Step 1 – Backward Test:**  
  Apply the model to the May 2025 GTA VI delay announcement (observed approximately −8% return) and to other GTA VI-related news (e.g., the 2023 trailer and 2024 hype period) to assess whether the model reproduces the direction and magnitude of observed reactions.

- **Validation Step 2 – External Expectations:**  
  Compare simulated GTA VI scenarios (bear, base, bull cases) against analyst forecasts and market-implied expectations to evaluate consistency with market beliefs.

---

### 3. Implementation

- Develop a modular and well-documented Python package (`src/eventstudy/`) with:
  - Clean code (PEP 8 compliance, `black`, `ruff`)
  - Unit tests
  - A command-line interface (CLI) for running analyses
- Visualize results using `matplotlib` and/or `plotly`, showing CAR by franchise, event type, and publisher.

---

## Expected Outcome

The project will deliver:
- A **reproducible event-study toolkit** for analyzing market reactions to video-game announcements.
- An analytical report explaining how releases and announcements influence publisher stock prices.
- Validation on recent real-world events and benchmarking against external market expectations.
- A **data-driven simulation of GTA VI’s potential market impact** and a methodology applicable to broader event-driven market research.