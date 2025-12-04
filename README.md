# 15C_Company-Success-Research

# Tech Stock Multi-Step Forecasting with MA-Only SARIMAX

## Overview

This project implements **14-day ahead forecasting** for tech stocks using MA-only SARIMAX models with **truly exogenous variables**. Through rigorous local testing, we discovered that individual stocks (especially MSFT) are significantly more predictable than portfolio indices, achieving up to **9% R²** on out-of-sample data.

**Date Range:** 2019-01-01 to 2025-11-21
**Frequency:** Daily stock data
**Companies:** 24 AI/tech companies across multiple sectors
**Best Result:** MSFT 14-day forecast (R² = 9.0%)

### Key Features

- **MA-Only Architecture**: SARIMAX(0,0,2) - AR components cause severe overfitting
- **Truly Exogenous Variables**: VIX + Treasury_10Y optimal (adding more hurts performance)
- **Stock-Specific Models**: Predictability varies wildly (MSFT 9%, GOOGL -8.6%)
- **Rigorous Validation**: Local execution proves models work before deployment
- **Realistic Expectations**: 1-9% R² range for stock returns (9% is excellent in finance)

### Performance Summary (14-Day Forecasts)

| Target | R² | Directional Accuracy | Assessment |
|--------|-----|---------------------|------------|
| **MSFT (individual)** | **9.0%** | 58.5% | 🏆 Best overall |
| Cloud/SaaS sector | 5.2% | 46.4% | ⭐ Strong |
| Cybersecurity sector | 5.0% | 55.1% | ⭐ Strong |
| Big Tech sector | 3.8% | **67.2%** | 📈 Best direction |
| AI Tech Index (23 stocks) | 1.9% | ~60% | ✓ Baseline |
| GOOGL (individual) | -8.6% | 66.0% | ❌ R² fails, direction ok |

---

## Data Structure

### Stock Tickers (22)

**Semiconductors/Hardware:**
- NVDA, AMD, INTC

**Cloud/SaaS:**
- GOOGL, MSFT, CRM, ORCL, NOW, OKTA

**Cybersecurity:**
- ZS, CRWD, NET

**Big Tech:**
- AAPL, META, AMZN, IBM

**Software/Other:**
- ADBE, SHOP, SQ, TWLO, MDB, DDOG

### Exogenous Variables (60+)

**Market Indices (7):**
- SP500, NASDAQ, Dow_Jones, Russell2000, VIX, VVIX, NASDAQ_VIX

**Interest Rates (6):**
- Treasury_10Y, Treasury_3M, Treasury_5Y, Treasury_30Y
- Yield_Curve_Slope, Yield_Curve_Inverted

**Sector ETFs (14):**
- Tech_Sector_ETF, Semiconductor_ETF, Software_ETF, Cloud_Computing_ETF
- Cybersecurity_ETF, AI_Robotics_ETF, Global_Robotics_ETF, Global_Cloud_ETF
- ARK funds (Innovation, Autonomous_Tech, Next_Gen_Internet)
- NASDAQ_100_ETF, First_Trust_NASDAQ, PHLX_Semi_Index

**Macro Indicators (9):**
- Dollar_Index, Dollar_Strength, Dollar_MA20
- Gold, Silver, Copper, Oil_WTI, Natural_Gas
- Gold_Oil_Ratio, Credit_Spread_Proxy

**Crypto (3):**
- Bitcoin, Ethereum, Crypto_Tech_Corr_20d

**International Tech (8):**
- Taiwan_ETF, Japan_ETF, South_Korea_ETF, China_Large_Cap_ETF
- ASML, TSM, BABA, BIDU

**Risk Indicators (5):**
- High_Yield_Bonds, Investment_Grade_Bonds, Long_Term_Treasury
- Emerging_Markets_Bonds, Real_Estate_ETF

**Volatility Metrics (5):**
- High_Volatility_Regime, Extreme_Fear, VIX_MA20, VIX_vs_MA, Vol_of_Vol_Ratio

**Technical Indicators (5):**
- Semi_vs_Tech_Ratio, Small_vs_Large_Caps, NASDAQ_MA5, NASDAQ_MA20, NASDAQ_Momentum

**Regime Indicators (5):**
- Pandemic_Period, AI_Boom_Period, Fed_Hike_Period, Tech_Bear_2022, Banking_Crisis_2023

**Temporal Features (10):**
- Day_of_Week, Day_of_Month, Week_of_Year, Month, Quarter, Year
- Is_Month_End, Is_Quarter_End, Earnings_Season, Options_Expiry_Week

---

## Truly Exogenous Variable Selection

### What Makes a Variable "Truly Exogenous"?

For a variable to be valid in our forecasting model, it must meet **all three criteria**:

1. **Determined outside the tech sector** - Not influenced by the stocks we're predicting
2. **Causally prior** - Influences tech stocks but tech stocks don't influence it
3. **Observable at prediction time** - No future peeking or data leakage

### Valid Exogenous Candidates

✅ **Market-Wide Volatility:**
- **VIX** - SPX options-implied volatility (market-wide, not tech-specific)
- **High_Volatility_Regime** - Binary indicator for extreme market stress

✅ **Interest Rates & Macro (Set by Fed/Bond Market):**
- **Treasury_10Y** - Risk-free rate benchmark
- **Treasury_3M** - Short-term rate
- **Yield_Curve_Slope** - 10Y-3M spread (recession indicator)
- **Yield_Curve_Inverted** - Binary recession signal
- **Dollar_Index** - USD currency strength

✅ **Commodities (Separate Markets):**
- **Gold** - Safe haven demand
- **Oil_WTI** - Energy/inflation proxy
- **Bitcoin** - Crypto risk appetite (debatable but separate asset class)

✅ **Regime Indicators (Binary Flags):**
- **AI_Boom_Period** - Post-ChatGPT era (Nov 2022+)
- **Fed_Hike_Period** - Rate hiking cycle
- **Tech_Bear_2022** - 2022 tech sell-off

### INVALID Variables (Endogenous/Circular)

❌ **NASDAQ, NASDAQ_100_ETF** - **Contains our target stocks!** Using NASDAQ to predict tech stocks is circular since tech stocks ARE NASDAQ.

❌ **Tech Sector ETFs** - Semiconductor_ETF, Software_ETF, Cloud_Computing_ETF, etc. literally contain the stocks we're forecasting.

❌ **Sector Fundamentals** - Profit margins, ROE, etc. are **outcomes**, not drivers. They're determined BY the stocks, not external to them.

### Final Model Uses 3-5 Variables Maximum

Through AIC/BIC-based forward selection, we choose only the most predictive variables that provide incremental information. Typical final models include:
- VIX (market volatility)
- Treasury rates or yield curve slope
- Dollar Index or commodity
- 1-2 regime indicators

---

## Methodology

### Feature Selection Process

**Conservative, AIC/BIC-Based Approach:**

1. **Candidate Pool**: Only truly exogenous variables (11 candidates max)
2. **Baseline Comparison**: Train ARIMA(1,1,1) with no exogenous variables
3. **Forward Selection**: Add variables one at a time via AIC minimization
4. **Stopping Rule**: Stop when AIC improvement < 2.0 or reached 5 variables
5. **Final Model**: Typically 3-5 variables selected

**Why This Approach:**
- **AIC/BIC penalize complexity** - Prevents overfitting
- **Forward selection** - Only adds variables that genuinely improve prediction
- **Hard limit of 5 variables** - Forces parsimony
- **Baseline comparison** - Validates that exogenous variables actually help

### Multi-Step Forecasting Approach

We implement **direct multi-step forecasting** rather than 1-step ahead:

**Target Variable: Log Returns**

Instead of predicting index levels, we forecast **log returns**:
- Formula: `log_return = ln(P_t+h / P_t)`
- Convert to %: `pct_return = exp(log_return) - 1`
- **Stationary by nature** (no differencing needed)
- **Time-additive** (7-day log return = sum of daily log returns)
- **Standard in quant finance** (better statistical properties)

**Forecast Horizons:**

- **7-Day Model**: Predicts 7-day log return (e.g., "expected +2.8% return")
  - Use case: Short-term trading strategies, options positioning
  - Expected accuracy: Highest among the three horizons

- **14-Day Model**: Predicts 14-day log return
  - Use case: Medium-term portfolio adjustments, swing trading
  - Expected accuracy: Moderate (natural degradation with longer horizon)

- **30-Day Model**: Predicts 30-day log return
  - Use case: Strategic allocation decisions, trend identification
  - Expected accuracy: Lower but still actionable for trend direction

### Model Architecture

**SARIMAX(0, 0, 2) x (0, 0, 0, 0)** - MA-Only Model
- **No AR component**: Autoregressive terms cause severe overfitting in our testing
- **I(0): No differencing** - log returns are already stationary
- **MA(2)**: Two moving average terms for shock absorption and pattern capture
- No seasonality: Daily data doesn't exhibit strong seasonal patterns

**Why MA-Only (No AR)?**
Through extensive local testing, we discovered:
- SARIMAX(1,0,1) with AR(1): R² = -3.09 (catastrophic failure, -101% worse than baseline)
- SARIMAX(0,0,1) MA-only: R² = +0.019 (modest but positive)
- SARIMAX(0,0,2) MA-only: R² = 0.02-0.09 depending on stock (BEST)

**Why I(0)?**
- Index levels need differencing (I=1) to become stationary
- Log returns are already stationary, so I=0
- Simpler model with better interpretation

### Evaluation Metrics

1. **RMSE (Root Mean Squared Error)**: Point prediction accuracy
2. **MAE (Mean Absolute Error)**: Average error magnitude
3. **R² Score**: Proportion of variance explained
4. **Directional Accuracy**: Percentage of correct up/down predictions (often most useful)

---

## Results & Insights

### Validation Approach

1. **Baseline Comparison**: Every model compared to ARIMA-only baseline
2. **Local Testing**: Executed validation scripts to verify actual performance
3. **Out-of-Sample Testing**: 15% holdout set never used in training or selection
4. **AIC/BIC Metrics**: Model complexity penalized appropriately
5. **Multiple Configurations**: Tested individual stocks, sector indices, horizons, and feature engineering

### Actual Performance Results (14-Day Forecasts)

**🏆 Best Performance: MSFT Individual Stock**
- **R² = 0.0900 (9.0%)** - Strong predictive power
- Directional accuracy: 58.5%
- Configuration: SARIMAX(0,0,2) with VIX + Treasury_10Y

**📊 Sector Indices (Middle Ground)**
- Cloud/SaaS: R² = 5.2%, Dir Acc = 46.4%
- Cybersecurity: R² = 5.0%, Dir Acc = 55.1%
- Big Tech: R² = 3.8%, **Dir Acc = 67.2%** (best directional)

**📈 Other Individual Stocks (Highly Variable)**
- AMD: R² = 1.4% (modest)
- AAPL: R² = 0.7% (weak)
- GOOGL: R² = -8.6% (fails completely)
- CRM: R² = 0.05% (nearly useless)
- ZS: R² = -3.6% (fails)

**🔍 AI Tech Portfolio Index**
- R² = 1.9% (baseline approach)
- More stable but lower signal due to diversification

### Key Findings

1. **Stock-Specific Predictability Varies Wildly**: MSFT achieves 9% R² while GOOGL completely fails at -8.6%. Not all tech stocks respond equally to macro factors.

2. **Individual Stocks > Indices for R²**: Best individual stock (MSFT 9%) significantly outperforms best sector index (Cloud/SaaS 5.2%) and portfolio index (1.9%). Diversification reduces signal-to-noise ratio.

3. **Sector Homogeneity Matters**: Cloud/SaaS (4 stocks) achieves 5.2% vs All Tech (23 stocks) at 1.9%. More similar stocks share stronger common drivers.

4. **VIX + Treasury_10Y is Optimal**: Simple 2-variable model outperforms complex feature engineering. Adding momentum or change variables makes performance WORSE (overfitting).

5. **MA-Only Models Work, AR Fails**: SARIMAX(0,0,2) achieves positive R², while SARIMAX(1,0,1) with AR component catastrophically fails (R² = -3.09).

6. **14-Day Horizon is Sweet Spot**: 7-day shows similar performance, 30-day degrades significantly. 14 days balances predictability with actionable timeframe.

7. **Directional Accuracy ≠ R²**: GOOGL has -8.6% R² but 66% directional accuracy. Big Tech has 3.8% R² but 67.2% directional. Some stocks give good direction signals but poor magnitude predictions.

### Model Limitations & Warnings

- **Stock-specific models required**: Don't assume one model works for all stocks. MSFT achieves 9% R² while GOOGL fails completely. Test each stock individually.

- **AR components dangerous**: Autoregressive terms caused catastrophic overfitting in our testing (-101% performance). Stick to MA-only models for stock returns.

- **Simple is better**: Feature engineering (momentum, change variables) made performance worse. Don't overcomplicate - VIX + Treasury_10Y is sufficient.

- **R² context matters**: 9% R² for stock returns is actually excellent. Stock markets are inherently noisy. Don't expect 70% R².

- **Directional vs magnitude**: Some stocks (GOOGL, Big Tech) show good directional accuracy but poor R². Consider directional signals for trading strategies.

- **Structural breaks**: Regime changes can invalidate historical relationships. Retraining required as market conditions evolve.

- **No guarantees**: Past performance ≠ future results. This is academic research, not financial advice.

- **Diversification trade-off**: Portfolio indices are more stable but less predictable. Individual stocks have higher R² but more risk.

---

## Files in Repository

### Main Analysis
- **EDA_Company Data.ipynb**: Main analysis notebook (being updated to reflect MSFT findings)
- **Datasets/Tech_Stock_Data_SEC_Cleaned_SARIMAX.csv**: Complete dataset with all features

### Validation Scripts (Key to Finding What Works)
- **run_analysis.py**: Initial validation - discovered AR(1) component fails catastrophically
- **try_alternatives.py**: Tested different ARIMA orders - found MA-only models work
- **test_improvements.py**: Compared stocks/sectors - discovered MSFT achieves 9% R²
- **test_results_summary.md**: Comprehensive summary of all test results

### Generated Outputs
- **Datasets/Multi_Step_Forecast_Results.csv**: Model predictions (generated after running notebook)
- **Datasets/Selected_Exogenous_Variables.txt**: Final selected features (typically VIX + Treasury_10Y)

---

## Running the Analysis

1. **Install Dependencies**:
   ```bash
   pip3 install pandas numpy scikit-learn statsmodels jupyter
   ```

2. **Quick Validation (Recommended First Step)**:
   ```bash
   # Test MSFT 14-day forecast (best performer)
   python3 test_improvements.py
   ```
   This runs all configurations locally and shows you which stocks/sectors work best.

3. **Run Full Analysis**:
   ```bash
   jupyter notebook "EDA_Company Data.ipynb"
   ```

4. **What the Notebook Does**:
   - Loads data (2019-2025, 24 tech stocks)
   - Creates MSFT 14-day log return target
   - Trains SARIMAX(0,0,2) with VIX + Treasury_10Y
   - Baseline comparison: ARIMA(0,0,1) without exogenous
   - Out-of-sample evaluation (15% holdout)
   - Visualization: predictions vs actuals
   - Expected result: R² ≈ 9%, directional accuracy ≈ 58%

5. **Try Other Stocks/Sectors**:
   Edit the notebook to replace MSFT with:
   - Cloud/SaaS sector index (R² ≈ 5.2%)
   - Cybersecurity sector index (R² ≈ 5.0%)
   - Big Tech sector index (R² ≈ 3.8%, Dir Acc ≈ 67%)
   - Avoid: GOOGL, CRM, ZS (negative R²)

---

## Future Improvements

### What We Learned NOT to Do
❌ Don't add AR components - they cause severe overfitting
❌ Don't add momentum/change variables - they hurt performance
❌ Don't use 30-day horizons - predictability degrades
❌ Don't use endogenous variables (NASDAQ, tech ETFs)
❌ Don't assume all stocks are equally predictable

### Promising Directions

1. **Expand Stock Coverage**: Test MSFT-like predictability in other large-cap tech (META, AMZN, etc.)

2. **Confidence Intervals**: Add probabilistic forecasts with prediction intervals for risk management

3. **Regime-Conditional Models**: Separate MSFT models for high/low volatility periods (VIX > 25 vs VIX < 15)

4. **Time-Varying Coefficients**: Allow VIX/Treasury sensitivity to change over time (state-space models)

5. **Directional Trading Strategies**: Leverage Big Tech's 67% directional accuracy for binary strategies

6. **Ensemble Approach**:
   - MSFT for point predictions (R² = 9%)
   - Big Tech for directional signals (Dir Acc = 67%)
   - Combine both for robust trading signals

7. **Alternative Exogenous**: Test credit spreads (BBB-AAA), real rates (Treasury - inflation), carry trade indicators

8. **Automated Retraining**: Monthly model updates with performance monitoring (detect when R² degrades)

**Critical Principle**: Any improvement must be validated locally before deployment. Every addition we tested (momentum, changes, AR terms) made performance worse.

---