#!/usr/bin/env python3
"""
Tech Stock Forecasting Dashboard
Interactive Streamlit app for MSFT 14-day forecasting with MA-only SARIMAX
"""
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Tech Stock Forecasting",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Tech Stock 14-Day Forecasting Dashboard")
st.markdown("**MA-Only SARIMAX with Truly Exogenous Variables**")

# Sidebar configuration
st.sidebar.header("⚙️ Model Configuration")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Datasets/Tech_Stock_Data_SEC_Cleaned_SARIMAX.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df

try:
    df = load_data()
    st.sidebar.success(f"✓ Data loaded: {df.shape[0]} rows")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Target selection
st.sidebar.subheader("1. Select Target")
target_type = st.sidebar.radio(
    "Target Type:",
    ["Individual Stock", "Sector Index", "Portfolio Index"]
)

if target_type == "Individual Stock":
    stock_options = {
        '🏆 MSFT (Best - R²=9.0%)': 'MSFT',
        'AMD (Modest - R²=1.4%)': 'AMD',
        'AAPL (Weak - R²=0.7%)': 'AAPL',
        'GOOGL (Fails - R²=-8.6%)': 'GOOGL',
        'CRM (Fails - R²=0.05%)': 'CRM',
        'ZS (Fails - R²=-3.6%)': 'ZS'
    }
    selected_stock = st.sidebar.selectbox("Stock:", list(stock_options.keys()))
    target_name = stock_options[selected_stock]

elif target_type == "Sector Index":
    sector_options = {
        'Cloud/SaaS (R²=5.2%)': ['CRM', 'ORCL', 'NOW', 'OKTA'],
        'Cybersecurity (R²=5.0%)': ['ZS', 'CRWD', 'PANW'],
        'Big Tech (R²=3.8%, Dir=67%)': ['GOOGL', 'MSFT', 'AAPL', 'META', 'AMZN']
    }
    selected_sector = st.sidebar.selectbox("Sector:", list(sector_options.keys()))
    target_name = selected_sector
else:
    target_name = "AI Tech Index (All 23 stocks)"

# Exogenous variables
st.sidebar.subheader("2. Exogenous Variables")
exog_options = ['VIX', 'Treasury_10Y', 'Yield_Curve_Slope', 'Dollar_Index',
                'AI_Boom_Period', 'Fed_Hike_Period']
available_exog = [v for v in exog_options if v in df.columns]

selected_exog = st.sidebar.multiselect(
    "Select variables:",
    available_exog,
    default=['VIX', 'Treasury_10Y']
)

if not selected_exog:
    st.sidebar.error("⚠️ Select at least one exogenous variable")
    st.stop()

# Model parameters
st.sidebar.subheader("3. Model Parameters")
ma_order = st.sidebar.slider("MA order (q):", 1, 3, 2)
train_split = st.sidebar.slider("Train/Test split:", 0.70, 0.90, 0.85, 0.05)

# Run button
run_model = st.sidebar.button("🚀 Run Forecast", type="primary")

# Main content
if run_model:
    with st.spinner("Training model..."):
        try:
            # Prepare target
            if target_type == "Individual Stock":
                stock_norm = df[target_name] / df[target_name].iloc[0] * 100
                logret_14d = np.log(stock_norm.shift(-14) / stock_norm)
                y = logret_14d.dropna()

            elif target_type == "Sector Index":
                stocks = sector_options[selected_sector]
                available_stocks = [s for s in stocks if s in df.columns]
                normalized = df[available_stocks].div(df[available_stocks].iloc[0]) * 100
                sector_index = normalized.mean(axis=1)
                logret_14d = np.log(sector_index.shift(-14) / sector_index)
                y = logret_14d.dropna()

            else:  # Portfolio Index
                stocks = ['NVDA', 'AMD', 'INTC', 'GOOGL', 'MSFT', 'AAPL', 'META', 'AMZN',
                         'CRM', 'ORCL', 'NOW', 'OKTA', 'ZS', 'CRWD', 'PANW',
                         'ADBE', 'SHOP', 'TWLO', 'MDB', 'DDOG', 'NET', 'PYPL', 'ANET']
                available_stocks = [s for s in stocks if s in df.columns]
                normalized = df[available_stocks].div(df[available_stocks].iloc[0]) * 100
                portfolio_index = normalized.mean(axis=1)
                logret_14d = np.log(portfolio_index.shift(-14) / portfolio_index)
                y = logret_14d.dropna()

            # Prepare exogenous
            X = df[selected_exog].loc[y.index].copy()
            X = X.fillna(method='ffill').fillna(method='bfill')

            # Train-test split
            train_size = int(len(y) * train_split)
            y_train, y_test = y[:train_size], y[train_size:]
            X_train, X_test = X[:train_size], X[train_size:]

            # Baseline ARIMA
            baseline = ARIMA(y_train, order=(0, 0, 1)).fit()
            baseline_pred = baseline.forecast(steps=len(y_test))
            baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
            baseline_r2 = r2_score(y_test, baseline_pred)

            # SARIMAX with exogenous
            model = SARIMAX(y_train, exog=X_train,
                           order=(0, 0, ma_order), seasonal_order=(0, 0, 0, 0),
                           enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False, maxiter=200)
            predictions = fitted.forecast(steps=len(y_test), exog=X_test)

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            actual_dir = np.sign(y_test)
            pred_dir = np.sign(predictions)
            dir_acc = (actual_dir == pred_dir).sum() / len(actual_dir)

            # Convert to percentage returns
            actual_pct = (np.exp(y_test) - 1) * 100
            pred_pct = (np.exp(predictions) - 1) * 100

            # Display results
            st.success("✓ Model trained successfully!")

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "R² Score",
                    f"{r2:.4f}",
                    f"{(r2-baseline_r2):.4f} vs baseline",
                    delta_color="normal"
                )

            with col2:
                st.metric(
                    "RMSE",
                    f"{rmse:.5f}",
                    f"{((baseline_rmse-rmse)/baseline_rmse*100):.1f}% improvement"
                )

            with col3:
                st.metric(
                    "Directional Accuracy",
                    f"{dir_acc:.1%}",
                    "% correct direction"
                )

            with col4:
                st.metric(
                    "Test Samples",
                    f"{len(y_test)}",
                    f"{len(y_test)/len(y)*100:.1f}% of data"
                )

            # Performance interpretation
            if r2 > 0.07:
                st.success(f"🏆 **Excellent Performance!** R² = {r2:.1%} exceeds 7% target")
            elif r2 > 0.03:
                st.info(f"⭐ **Good Performance** - R² = {r2:.1%} is solid for stock returns")
            elif r2 > 0:
                st.warning(f"⚠️ **Modest Performance** - R² = {r2:.1%} beats baseline but weak signal")
            else:
                st.error(f"❌ **Model Fails** - Negative R² = {r2:.1%}. Try different target/configuration")

            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast Chart", "📈 Scatter Plot", "📋 Predictions Table", "🔍 Model Details"])

            with tab1:
                st.subheader("Out-of-Sample Predictions")

                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(y_test.index, actual_pct, 'o-', label='Actual', alpha=0.7, markersize=5)
                ax.plot(y_test.index, pred_pct, 's-', label='Predicted', alpha=0.7, markersize=5)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
                ax.set_title(f'{target_name} 14-Day Returns: Actual vs Predicted (R² = {r2:.4f})',
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('14-Day Return (%)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            with tab2:
                st.subheader("Actual vs Predicted Scatter")

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(actual_pct, pred_pct, alpha=0.6, s=50)

                # Perfect prediction line
                min_val = min(actual_pct.min(), pred_pct.min())
                max_val = max(actual_pct.max(), pred_pct.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Prediction')

                ax.set_title(f'Actual vs Predicted Returns (R² = {r2:.4f})', fontsize=14, fontweight='bold')
                ax.set_xlabel('Actual 14-Day Return (%)')
                ax.set_ylabel('Predicted 14-Day Return (%)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

            with tab3:
                st.subheader("Prediction Details")

                results_df = pd.DataFrame({
                    'Date': y_test.index,
                    'Actual (%)': actual_pct.values,
                    'Predicted (%)': pred_pct.values,
                    'Error (%)': (actual_pct - pred_pct).values,
                    'Direction': ['✓' if np.sign(a) == np.sign(p) else '✗'
                                 for a, p in zip(actual_pct, pred_pct)]
                })

                st.dataframe(
                    results_df.style.format({
                        'Actual (%)': '{:.2f}',
                        'Predicted (%)': '{:.2f}',
                        'Error (%)': '{:.2f}'
                    }),
                    height=400
                )

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name=f"{target_name}_predictions.csv",
                    mime="text/csv"
                )

            with tab4:
                st.subheader("Model Summary")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Configuration:**")
                    st.write(f"- Target: {target_name}")
                    st.write(f"- Model: SARIMAX(0, 0, {ma_order})")
                    st.write(f"- Exogenous: {', '.join(selected_exog)}")
                    st.write(f"- Train samples: {len(y_train)}")
                    st.write(f"- Test samples: {len(y_test)}")

                with col2:
                    st.markdown("**Performance Metrics:**")
                    st.write(f"- Baseline R²: {baseline_r2:.4f}")
                    st.write(f"- SARIMAX R²: {r2:.4f}")
                    st.write(f"- Improvement: {((baseline_rmse-rmse)/baseline_rmse*100):.1f}%")
                    st.write(f"- AIC: {fitted.aic:.2f}")
                    st.write(f"- BIC: {fitted.bic:.2f}")

                st.markdown("**Model Coefficients:**")
                st.dataframe(
                    fitted.summary().tables[1],
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
            st.exception(e)

else:
    # Initial page content
    st.info("👈 Configure settings in the sidebar and click **Run Forecast** to begin")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Best Performers (14-Day Forecast)")

        best_df = pd.DataFrame({
            'Target': ['MSFT', 'Cloud/SaaS', 'Cybersecurity', 'Big Tech', 'AMD'],
            'R²': ['9.0%', '5.2%', '5.0%', '3.8%', '1.4%'],
            'Dir Acc': ['58.5%', '46.4%', '55.1%', '67.2%', '54.0%'],
            'Assessment': ['🏆 Best overall', '⭐ Strong', '⭐ Strong', '📈 Best direction', '⚠️ Modest']
        })

        st.dataframe(best_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("❌ Poor Performers")

        poor_df = pd.DataFrame({
            'Target': ['GOOGL', 'ZS', 'CRM'],
            'R²': ['-8.6%', '-3.6%', '0.05%'],
            'Dir Acc': ['66.0%', '52.1%', '38.9%'],
            'Note': ['R² fails, direction ok', 'Both metrics fail', 'Nearly useless']
        })

        st.dataframe(poor_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("📚 Key Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **✅ What Works:**
        - MA-only models (no AR)
        - VIX + Treasury_10Y
        - Individual stocks (MSFT)
        - 14-day horizon
        """)

    with col2:
        st.markdown("""
        **❌ What Fails:**
        - AR components (-101% perf)
        - Momentum variables
        - 30-day horizons
        - Endogenous variables
        """)

    with col3:
        st.markdown("""
        **💡 Important Notes:**
        - 9% R² is excellent for stocks
        - Test each stock individually
        - Simple > complex
        - Direction ≠ magnitude
        """)

# Footer
st.markdown("---")
st.caption("Tech Stock Forecasting Dashboard | MA-Only SARIMAX with Truly Exogenous Variables")
