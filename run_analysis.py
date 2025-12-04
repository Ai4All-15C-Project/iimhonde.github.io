#!/usr/bin/env python3
"""
Quick validation script - runs key parts of the analysis
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("RUNNING ANALYSIS - KEY SECTIONS ONLY")
print("="*70)

# 1. Load data
print("\n1. Loading data...")
df = pd.read_csv('Datasets/Tech_Stock_Data_SEC_Cleaned_SARIMAX.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
print(f"   Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Date range: {df.index.min()} to {df.index.max()}")

# 2. Create AI Tech Index
print("\n2. Creating AI Tech Index...")
stocks = ['NVDA', 'AMD', 'INTC', 'GOOGL', 'MSFT', 'AAPL', 'META', 'AMZN',
          'CRM', 'ORCL', 'NOW', 'OKTA', 'ZS', 'CRWD', 'PANW',
          'ADBE', 'SHOP', 'TWLO', 'MDB', 'DDOG', 'NET', 'PYPL', 'ANET']
normalized = df[stocks].div(df[stocks].iloc[0]) * 100
df['AI_Tech_Index'] = normalized.mean(axis=1)
print(f"   Index range: {df['AI_Tech_Index'].min():.2f} to {df['AI_Tech_Index'].max():.2f}")

# 3. Compute log returns
print("\n3. Computing 7-day log returns...")
df['logret_7d'] = np.log(df['AI_Tech_Index'].shift(-7) / df['AI_Tech_Index'])
print(f"   Mean: {df['logret_7d'].mean():.6f}")
print(f"   Std: {df['logret_7d'].std():.6f}")
print(f"   Valid observations: {df['logret_7d'].notna().sum()}")

# 4. Select exogenous variables
print("\n4. Selecting exogenous variables...")
exog_candidates = ['VIX', 'Treasury_10Y', 'Yield_Curve_Slope', 'Dollar_Index',
                   'Bitcoin', 'AI_Boom_Period', 'Fed_Hike_Period']
available = [v for v in exog_candidates if v in df.columns]
print(f"   Available: {available}")

X_full = df[available].copy()
X_full = X_full.fillna(method='ffill').fillna(method='bfill')

# 5. Train-test split
print("\n5. Train-test split...")
y = df['logret_7d'].dropna()
X = X_full.loc[y.index]
train_size = int(len(y) * 0.85)
y_train, y_test = y[:train_size], y[train_size:]
X_train, X_test = X[:train_size], X[train_size:]
print(f"   Train: {len(y_train)}, Test: {len(y_test)}")

# 6. Baseline ARIMA
print("\n6. Training baseline ARIMA(1,0,1)...")
baseline = ARIMA(y_train, order=(1, 0, 1)).fit()
baseline_pred = baseline.forecast(steps=len(y_test))
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = r2_score(y_test, baseline_pred)
print(f"   Baseline RMSE: {baseline_rmse:.6f}")
print(f"   Baseline R²: {baseline_r2:.4f}")
print(f"   Baseline AIC: {baseline.aic:.2f}")

# 7. Forward selection (simplified - top 3 only)
print("\n7. Testing individual variables...")
best_var = None
best_aic = baseline.aic
for var in available[:5]:  # Test first 5
    try:
        model = SARIMAX(y_train, exog=X_train[[var]],
                       order=(1, 0, 1), seasonal_order=(0, 0, 0, 0),
                       enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=100)
        if fitted.aic < best_aic:
            best_aic = fitted.aic
            best_var = var
        print(f"   {var}: AIC={fitted.aic:.2f} (improvement: {baseline.aic - fitted.aic:.2f})")
    except Exception as e:
        print(f"   {var}: FAILED - {str(e)[:50]}")

selected = [best_var] if best_var else []
print(f"\n   Selected: {selected}")

# 8. Final SARIMAX model
if selected:
    print("\n8. Training SARIMAX with selected variables...")
    model = SARIMAX(y_train, exog=X_train[selected],
                   order=(1, 0, 1), seasonal_order=(0, 0, 0, 0),
                   enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200)
    predictions = fitted.forecast(steps=len(y_test), exog=X_test[selected])

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Directional accuracy
    actual_dir = np.sign(y_test)
    pred_dir = np.sign(predictions)
    dir_acc = (actual_dir == pred_dir).sum() / len(actual_dir)

    print(f"   SARIMAX RMSE: {rmse:.6f} (baseline: {baseline_rmse:.6f})")
    print(f"   SARIMAX R²: {r2:.4f} (baseline: {baseline_r2:.4f})")
    print(f"   Directional Accuracy: {dir_acc:.2%}")
    print(f"   Improvement: {((baseline_rmse - rmse) / baseline_rmse * 100):.1f}%")

    # Sample predictions
    print("\n9. Sample predictions (first 10):")
    print("   Date              Actual    Predicted  (% returns)")
    for i in range(min(10, len(y_test))):
        actual_pct = (np.exp(y_test.iloc[i]) - 1) * 100
        pred_pct = (np.exp(predictions.iloc[i]) - 1) * 100
        date_str = y_test.index[i].strftime('%Y-%m-%d')
        print(f"   {date_str}  {actual_pct:7.2f}%  {pred_pct:7.2f}%")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
