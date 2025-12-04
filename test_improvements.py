#!/usr/bin/env python3
"""
Test different improvements to boost R² above 0.02
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TESTING IMPROVEMENTS TO BOOST R²")
print("="*70)

# Load data
df = pd.read_csv('Datasets/Tech_Stock_Data_SEC_Cleaned_SARIMAX.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Stocks to test (excluding NVDA)
test_stocks = ['AMD', 'GOOGL', 'MSFT', 'AAPL', 'CRM', 'ZS']

# Base exogenous
base_exog = ['VIX', 'Treasury_10Y', 'Yield_Curve_Slope', 'AI_Boom_Period']
available_exog = [v for v in base_exog if v in df.columns]
X_base = df[available_exog].fillna(method='ffill').fillna(method='bfill')

results = []

# ============================================================================
# TEST 1: Individual Stocks (14-day horizon)
# ============================================================================
print("\n" + "="*70)
print("TEST 1: Individual Stocks (14-day log returns)")
print("="*70)

for stock in test_stocks:
    if stock not in df.columns:
        continue

    print(f"\n{stock}:")

    # Normalize to base 100
    stock_norm = df[stock] / df[stock].iloc[0] * 100

    # 14-day log returns
    logret_14d = np.log(stock_norm.shift(-14) / stock_norm)

    y = logret_14d.dropna()
    X = X_base.loc[y.index]

    train_size = int(len(y) * 0.85)
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]

    # Baseline
    baseline = ARIMA(y_train, order=(0, 0, 1)).fit()
    baseline_pred = baseline.forecast(steps=len(y_test))
    baseline_r2 = r2_score(y_test, baseline_pred)

    # With exogenous (MA(2) model)
    try:
        model = SARIMAX(y_train, exog=X_train[['VIX', 'Treasury_10Y']],
                       order=(0, 0, 2), seasonal_order=(0, 0, 0, 0),
                       enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=200)
        pred = fitted.forecast(steps=len(y_test), exog=X_test[['VIX', 'Treasury_10Y']])
        r2 = r2_score(y_test, pred)

        dir_acc = (np.sign(y_test) == np.sign(pred)).sum() / len(y_test)

        print(f"  Baseline R²: {baseline_r2:.4f}")
        print(f"  SARIMAX R²:  {r2:.4f}")
        print(f"  Dir Acc:     {dir_acc:.2%}")

        results.append({
            'Test': 'Individual Stock',
            'Target': stock,
            'Horizon': '14d',
            'R2': r2,
            'Dir_Acc': dir_acc
        })
    except Exception as e:
        print(f"  FAILED: {str(e)[:50]}")

# ============================================================================
# TEST 2: Sector Indices (not just equal-weighted)
# ============================================================================
print("\n" + "="*70)
print("TEST 2: Sector-Specific Indices (14-day)")
print("="*70)

sectors = {
    'Big_Tech': ['GOOGL', 'MSFT', 'AAPL', 'META', 'AMZN'],
    'Cloud_SaaS': ['CRM', 'ORCL', 'NOW', 'OKTA'],
    'Cybersecurity': ['ZS', 'CRWD', 'PANW']
}

for sector_name, stocks in sectors.items():
    available_stocks = [s for s in stocks if s in df.columns]
    if len(available_stocks) < 2:
        continue

    print(f"\n{sector_name}:")

    # Equal-weighted sector index
    normalized = df[available_stocks].div(df[available_stocks].iloc[0]) * 100
    sector_index = normalized.mean(axis=1)

    logret_14d = np.log(sector_index.shift(-14) / sector_index)
    y = logret_14d.dropna()
    X = X_base.loc[y.index]

    train_size = int(len(y) * 0.85)
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]

    try:
        baseline = ARIMA(y_train, order=(0, 0, 1)).fit()
        baseline_pred = baseline.forecast(steps=len(y_test))
        baseline_r2 = r2_score(y_test, baseline_pred)

        model = SARIMAX(y_train, exog=X_train[['VIX', 'Treasury_10Y']],
                       order=(0, 0, 2), seasonal_order=(0, 0, 0, 0),
                       enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=200)
        pred = fitted.forecast(steps=len(y_test), exog=X_test[['VIX', 'Treasury_10Y']])
        r2 = r2_score(y_test, pred)

        dir_acc = (np.sign(y_test) == np.sign(pred)).sum() / len(y_test)

        print(f"  Baseline R²: {baseline_r2:.4f}")
        print(f"  SARIMAX R²:  {r2:.4f}")
        print(f"  Dir Acc:     {dir_acc:.2%}")

        results.append({
            'Test': 'Sector Index',
            'Target': sector_name,
            'Horizon': '14d',
            'R2': r2,
            'Dir_Acc': dir_acc
        })
    except Exception as e:
        print(f"  FAILED: {str(e)[:50]}")

# ============================================================================
# TEST 3: With Momentum Variables (GOOGL as example)
# ============================================================================
print("\n" + "="*70)
print("TEST 3: Adding Momentum Variables (GOOGL, 14-day)")
print("="*70)

stock = 'GOOGL'
stock_norm = df[stock] / df[stock].iloc[0] * 100

# Create momentum features
df['GOOGL_ret_5d'] = np.log(stock_norm / stock_norm.shift(5))
df['GOOGL_ret_20d'] = np.log(stock_norm / stock_norm.shift(20))

logret_14d = np.log(stock_norm.shift(-14) / stock_norm)
y = logret_14d.dropna()

# Enhanced exogenous with momentum
X_enhanced = df[['VIX', 'Treasury_10Y', 'GOOGL_ret_5d', 'GOOGL_ret_20d']].loc[y.index]
X_enhanced = X_enhanced.fillna(method='ffill').fillna(method='bfill')

train_size = int(len(y) * 0.85)
y_train, y_test = y[:train_size], y[train_size:]
X_train, X_test = X_enhanced[:train_size], X_enhanced[train_size:]

# Baseline
baseline = ARIMA(y_train, order=(0, 0, 1)).fit()
baseline_pred = baseline.forecast(steps=len(y_test))
baseline_r2 = r2_score(y_test, baseline_pred)

# Without momentum
try:
    model1 = SARIMAX(y_train, exog=X_train[['VIX', 'Treasury_10Y']],
                    order=(0, 0, 2), seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted1 = model1.fit(disp=False, maxiter=200)
    pred1 = fitted1.forecast(steps=len(y_test), exog=X_test[['VIX', 'Treasury_10Y']])
    r2_no_momentum = r2_score(y_test, pred1)

    print(f"\nWithout momentum:")
    print(f"  R²: {r2_no_momentum:.4f}")
except Exception as e:
    print(f"Without momentum FAILED: {str(e)[:50]}")
    r2_no_momentum = None

# With momentum
try:
    model2 = SARIMAX(y_train, exog=X_train,
                    order=(0, 0, 2), seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted2 = model2.fit(disp=False, maxiter=200)
    pred2 = fitted2.forecast(steps=len(y_test), exog=X_test)
    r2_with_momentum = r2_score(y_test, pred2)

    dir_acc = (np.sign(y_test) == np.sign(pred2)).sum() / len(y_test)

    print(f"\nWith momentum (ret_5d, ret_20d):")
    print(f"  R²: {r2_with_momentum:.4f}")
    print(f"  Dir Acc: {dir_acc:.2%}")

    if r2_no_momentum:
        improvement = r2_with_momentum - r2_no_momentum
        print(f"  Momentum gain: {improvement:.4f}")

    results.append({
        'Test': 'With Momentum',
        'Target': 'GOOGL',
        'Horizon': '14d',
        'R2': r2_with_momentum,
        'Dir_Acc': dir_acc
    })
except Exception as e:
    print(f"With momentum FAILED: {str(e)[:50]}")

# ============================================================================
# TEST 4: Change Variables (MSFT as example)
# ============================================================================
print("\n" + "="*70)
print("TEST 4: Change Variables (MSFT, 14-day)")
print("="*70)

stock = 'MSFT'
stock_norm = df[stock] / df[stock].iloc[0] * 100

# Create change features
df['VIX_change'] = df['VIX'].diff()
df['Rate_change'] = df['Treasury_10Y'].diff()

logret_14d = np.log(stock_norm.shift(-14) / stock_norm)
y = logret_14d.dropna()

X_changes = df[['VIX', 'VIX_change', 'Treasury_10Y', 'Rate_change']].loc[y.index]
X_changes = X_changes.fillna(method='ffill').fillna(method='bfill')

train_size = int(len(y) * 0.85)
y_train, y_test = y[:train_size], y[train_size:]
X_train, X_test = X_changes[:train_size], X_changes[train_size:]

# Baseline
baseline = ARIMA(y_train, order=(0, 0, 1)).fit()
baseline_pred = baseline.forecast(steps=len(y_test))
baseline_r2 = r2_score(y_test, baseline_pred)

try:
    # Levels only
    model1 = SARIMAX(y_train, exog=X_train[['VIX', 'Treasury_10Y']],
                    order=(0, 0, 2), seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted1 = model1.fit(disp=False, maxiter=200)
    pred1 = fitted1.forecast(steps=len(y_test), exog=X_test[['VIX', 'Treasury_10Y']])
    r2_levels = r2_score(y_test, pred1)

    print(f"\nLevels only:")
    print(f"  R²: {r2_levels:.4f}")

    # Levels + changes
    model2 = SARIMAX(y_train, exog=X_train,
                    order=(0, 0, 2), seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted2 = model2.fit(disp=False, maxiter=200)
    pred2 = fitted2.forecast(steps=len(y_test), exog=X_test)
    r2_changes = r2_score(y_test, pred2)

    dir_acc = (np.sign(y_test) == np.sign(pred2)).sum() / len(y_test)

    print(f"\nLevels + changes:")
    print(f"  R²: {r2_changes:.4f}")
    print(f"  Dir Acc: {dir_acc:.2%}")
    print(f"  Change gain: {r2_changes - r2_levels:.4f}")

    results.append({
        'Test': 'With Changes',
        'Target': 'MSFT',
        'Horizon': '14d',
        'R2': r2_changes,
        'Dir_Acc': dir_acc
    })
except Exception as e:
    print(f"FAILED: {str(e)[:50]}")

# ============================================================================
# TEST 5: Longer Horizon (30-day for AAPL)
# ============================================================================
print("\n" + "="*70)
print("TEST 5: 30-Day Horizon (AAPL)")
print("="*70)

stock = 'AAPL'
stock_norm = df[stock] / df[stock].iloc[0] * 100
logret_30d = np.log(stock_norm.shift(-30) / stock_norm)

y = logret_30d.dropna()
X = X_base.loc[y.index]

train_size = int(len(y) * 0.85)
y_train, y_test = y[:train_size], y[train_size:]
X_train, X_test = X[:train_size], X[train_size:]

try:
    baseline = ARIMA(y_train, order=(0, 0, 1)).fit()
    baseline_pred = baseline.forecast(steps=len(y_test))
    baseline_r2 = r2_score(y_test, baseline_pred)

    model = SARIMAX(y_train, exog=X_train[['VIX', 'Treasury_10Y']],
                   order=(0, 0, 2), seasonal_order=(0, 0, 0, 0),
                   enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200)
    pred = fitted.forecast(steps=len(y_test), exog=X_test[['VIX', 'Treasury_10Y']])
    r2 = r2_score(y_test, pred)

    dir_acc = (np.sign(y_test) == np.sign(pred)).sum() / len(y_test)

    print(f"Baseline R²: {baseline_r2:.4f}")
    print(f"SARIMAX R²:  {r2:.4f}")
    print(f"Dir Acc:     {dir_acc:.2%}")

    results.append({
        'Test': '30-day horizon',
        'Target': 'AAPL',
        'Horizon': '30d',
        'R2': r2,
        'Dir_Acc': dir_acc
    })
except Exception as e:
    print(f"FAILED: {str(e)[:50]}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY OF ALL TESTS")
print("="*70)

if results:
    results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
    print(f"\nTop 10 Results by R²:")
    print(results_df.head(10).to_string(index=False))

    print(f"\n\nBest R²: {results_df['R2'].max():.4f}")
    best = results_df.iloc[0]
    print(f"Configuration: {best['Test']}, {best['Target']}, {best['Horizon']}")
else:
    print("No successful results")

print("\n" + "="*70)
