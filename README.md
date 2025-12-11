# 15C Company Success Research - AI Tech Sector Forecasting

## Project Title

**Forecasting AI Tech Sector Performance Using SARIMAX Time Series Analysis with Exogenous Market Factors**

Developed an advanced SARIMAX forecasting model that predicts the AI Technology Index (normalized average of 24 leading AI and tech companies) by leveraging truly exogenous market variables, SEC fundamentals, and sophisticated time series methodologies, all within the AI4ALL Ignite accelerator program.

---

## Problem Statement

The artificial intelligence sector has emerged as a critical economic driver, but predicting AI tech company performance remains challenging due to market volatility, macroeconomic sensitivity, and regime shifts. Traditional models often suffer from circular reasoning when using market indices (SP500, NASDAQ) that contain the same stocks being predicted, leading to inflated accuracy and poor generalization. This project addresses the need for a rigorous, methodologically sound forecasting approach that uses only truly exogenous variables, ensuring legitimate predictive power without data leakage. By successfully forecasting the AI tech sector, investors, analysts, and portfolio managers gain insight into AI-driven market trends during a period of unprecedented technological change.

---

## Key Results

- Successfully engineered a normalized AI Tech Index from 24 major AI and tech companies spanning semiconductors (NVDA, AMD, INTC), cloud services (MSFT, GOOGL), cybersecurity (CRWD, OKTA, PANW), and software (ADBE, MDB, DDOG)

- Identified and validated truly exogenous predictors (VIX, Treasury yields, sector profit margins) while explicitly rejecting endogenous market indices to eliminate circular reasoning in forecasting

- Achieved rolling 1-step-ahead forecasts with R² of approximately 0.97 on daily predictions, demonstrating strong short-term predictive ability

- Implemented multicollinearity diagnostics using VIF analysis and performed comprehensive residual diagnostics (Ljung-Box, Jarque-Bera, heteroskedasticity tests) confirming model adequacy

- Detected market anomalies and regime shifts using Isolation Forest, identifying unusual trading periods that corresponded with significant market events (2024-2025 AI boom, volatility spikes)

- Created production-ready data pipeline cleaning 6+ years of daily stock data (2019-2025), handling missing values, engineering lagged features, and generating rolling statistics

---

## Methodologies

We constructed a comprehensive SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) model to forecast the AI Tech Index:

**Data Engineering:** Collected daily prices for 24 tech companies from 2019-2025, removed weekends and market holidays, and created the AI Tech Index as an equal-weighted normalized average. Engineered over 60 exogenous features including market indices, interest rate curves, commodity prices, cryptocurrency correlations, SEC sector fundamentals, and binary regime indicators (pandemic period, AI boom, Fed rate hikes).

**Critical Methodological Decision:** Explicitly excluded SP500 and NASDAQ indices as predictors despite their high correlation, recognizing they contain the same stocks as our target variable. Instead, we selected truly exogenous variables using VIF analysis (variance inflation factor <5) to eliminate multicollinearity and ensure legitimate forecasting without circular reasoning.

**Model Development:** Compared SARIMAX with exogenous variables versus pure ARIMA using AIC/BIC criteria for model selection. Fitted SARIMAX(1,1,1) or SARIMAX(2,1,2) on lagged features (1-day lag for valid forecasting without data leakage). Implemented train-test split (90% training on 2019-2024, 10% testing on 2025) to evaluate generalization.

**Evaluation Strategy:** Employed rolling 1-step-ahead forecasts that simulate real-world usage by re-fitting the model each day with newly available data, achieving dramatically better performance than long-horizon forecasts. Calculated R², RMSE, and MAE metrics on both training and test sets to detect overfitting. Performed residual diagnostics including Ljung-Box autocorrelation test, Jarque-Bera normality test, and heteroskedasticity checks.

**Anomaly Detection:** Applied Isolation Forest algorithm to detect unusual prediction errors corresponding to market stress events, identifying approximately 5% of observations as anomalies.

---

## Data Sources

- **Yahoo Finance API** - Daily OHLCV data for 24 tech company stocks (2019-2025)
- **FRED (Federal Reserve Economic Data)** - Treasury yields (3M, 5Y, 10Y, 30Y), yield curve slope
- **CBOE (Chicago Board Options Exchange)** - VIX volatility index, VVIX
- **SEC EDGAR Database** - Sector-level fundamental metrics (profit margins, ROE, revenue growth)
- **Crypto APIs** - Bitcoin and Ethereum daily prices and correlations
- **ETF Providers** - Sector and thematic ETF prices (semiconductors, cloud computing, cybersecurity, AI robotics)

---

## Technologies Used

**Core Libraries:**
- Python 3.x
- pandas - Data manipulation and time series handling
- NumPy - Numerical computations
- statsmodels - SARIMAX model implementation, ACF/PACF analysis, residual diagnostics
- scikit-learn - Train-test splitting, VIF calculation, Isolation Forest anomaly detection
- matplotlib and seaborn - Data visualization and exploratory analysis
- yfinance - Yahoo Finance API wrapper for stock data

**Specialized Tools:**
- Jupyter Notebook - Interactive development and documentation
- Streamlit - Web app deployment and interactive dashboard
- plotly - Interactive visualizations for dashboard
- scipy - Statistical testing (Jarque-Bera, autocorrelation tests)

---

## Authors

This project was completed in collaboration with the AI4ALL Ignite cohort (15C):

- Primary Contributors - Ivie Imhonde, Shuo Wu, Mohammad El-Tawil
- Mentorship and Guidance - AI4ALL Faculty Advisors
- Repository - Ai4All-15C-Project Organization

For questions or collaboration inquiries, please refer to the main GitHub repository.

---