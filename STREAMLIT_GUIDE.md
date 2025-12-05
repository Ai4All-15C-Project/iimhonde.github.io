# Streamlit App Quick Start Guide

## 📊 Tech Stock Forecasting Dashboard

Interactive web application for exploring MSFT 14-day forecasting and testing different configurations.

## Installation

1. **Install dependencies** (if not already installed):
   ```bash
   pip3 install -r requirements.txt
   ```

   Or install streamlit specifically:
   ```bash
   pip3 install streamlit
   ```

2. **Verify installation**:
   ```bash
   streamlit --version
   ```

## Running the App

### Local Deployment

From the repository directory, run:

```bash
streamlit run streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Command Line Options

```bash
# Run on specific port
streamlit run streamlit_app.py --server.port 8080

# Run without opening browser
streamlit run streamlit_app.py --server.headless true

# Enable wide mode by default
streamlit run streamlit_app.py --server.enableXsrfProtection false
```

## Using the App

### 1. Select Target
Choose what you want to forecast:
- **Individual Stock**: MSFT (best), AMD, AAPL, GOOGL, CRM, ZS
- **Sector Index**: Cloud/SaaS, Cybersecurity, Big Tech
- **Portfolio Index**: All 23 tech stocks

### 2. Configure Exogenous Variables
Select which macro variables to use:
- **VIX** - Market volatility (recommended)
- **Treasury_10Y** - 10-year yield (recommended)
- **Yield_Curve_Slope** - Recession indicator
- **Dollar_Index** - USD strength
- **AI_Boom_Period** - Post-ChatGPT era
- **Fed_Hike_Period** - Rate hiking cycle

**Default (VIX + Treasury_10Y)** is optimal based on testing.

### 3. Model Parameters
- **MA order**: Number of moving average terms (default: 2)
- **Train/Test split**: Percentage for training (default: 85%)

### 4. Run Forecast
Click **🚀 Run Forecast** to train the model and see results.

## Features

### 📊 Forecast Chart
- Interactive time series plot
- Actual vs Predicted returns
- Out-of-sample performance

### 📈 Scatter Plot
- Correlation visualization
- Perfect prediction line for reference
- R² interpretation

### 📋 Predictions Table
- Detailed prediction breakdown
- Directional accuracy markers (✓/✗)
- Downloadable CSV export

### 🔍 Model Details
- Full model configuration
- Performance metrics comparison
- Coefficient estimates with p-values

## Expected Performance

Based on our testing:

| Target | Expected R² | Dir Accuracy | Notes |
|--------|-------------|--------------|-------|
| MSFT | ~9.0% | ~58% | 🏆 Best performer |
| Cloud/SaaS | ~5.2% | ~46% | ⭐ Strong |
| Cybersecurity | ~5.0% | ~55% | ⭐ Balanced |
| Big Tech | ~3.8% | ~67% | 📈 Best direction |
| GOOGL | -8.6% | ~66% | ❌ R² fails |

## Tips for Best Results

### ✅ Do:
- Use MSFT for best R² performance
- Stick with VIX + Treasury_10Y (simple is better)
- Try sector indices for more stable predictions
- Use Big Tech for directional trading signals

### ❌ Don't:
- Don't add too many exogenous variables (overfitting)
- Don't use GOOGL, CRM, or ZS (they fail)
- Don't expect 50%+ R² (9% is excellent for stocks)
- Don't enable AR components (they're hardcoded off for good reason)

## Customization

### Adding New Stocks

Edit the `stock_options` dictionary in `streamlit_app.py`:

```python
stock_options = {
    'Your Stock Name': 'TICKER',
    # ... existing stocks
}
```

### Adding New Exogenous Variables

If you have new variables in the dataset, add them to `exog_options`:

```python
exog_options = ['VIX', 'Treasury_10Y', 'YOUR_NEW_VAR']
```

### Changing Default Configuration

Modify the default values in sidebar widgets:

```python
selected_exog = st.sidebar.multiselect(
    "Select variables:",
    available_exog,
    default=['VIX', 'YOUR_FAVORITE_VAR']  # Change here
)
```

## Troubleshooting

### App won't start
```bash
# Check if streamlit is installed
pip3 show streamlit

# Reinstall if needed
pip3 install --upgrade streamlit
```

### Data not loading
- Verify `Datasets/Tech_Stock_Data_SEC_Cleaned_SARIMAX.csv` exists
- Check file path is correct relative to where you run streamlit

### Model errors
- Ensure you have at least one exogenous variable selected
- Try reducing MA order if convergence fails
- Check for NaN values in selected variables

### Performance issues
- Reduce train/test split for faster computation
- Use fewer exogenous variables
- Close other browser tabs

## Deployment Options

### Streamlit Cloud (Free)
1. Push repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and select `streamlit_app.py`
4. Deploy!

### Heroku
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port $PORT" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

## Advanced Usage

### API Mode (Programmatic)

You can import functions from the app:

```python
from streamlit_app import load_data, run_forecast

df = load_data()
results = run_forecast(
    df=df,
    target='MSFT',
    exog_vars=['VIX', 'Treasury_10Y'],
    ma_order=2
)
```

### Batch Processing

Run multiple configurations:

```python
stocks = ['MSFT', 'AMD', 'AAPL']
for stock in stocks:
    # ... configure and run
    # ... save results
```

## Performance Monitoring

The app displays:
- Real-time R² score
- RMSE improvement vs baseline
- Directional accuracy
- Model convergence status

Monitor these metrics to evaluate if your configuration is working.

## Support & Resources

- **Test Scripts**: Run `python3 test_improvements.py` to see all configurations
- **Documentation**: See `README.md` for methodology
- **Results Summary**: See `test_results_summary.md` for detailed findings
- **Standalone Script**: See `msft_forecast_best.py` for non-interactive version

## Example Session

```bash
# 1. Install
pip3 install streamlit

# 2. Run app
streamlit run streamlit_app.py

# 3. In browser:
#    - Select "MSFT (Best - R²=9.0%)"
#    - Keep default VIX + Treasury_10Y
#    - Click "Run Forecast"
#    - See R² ≈ 9% 🎉

# 4. Experiment:
#    - Try "Big Tech" sector for best directional accuracy
#    - Try "GOOGL" to see what failure looks like
#    - Add/remove exogenous variables to see impact
```

## Notes

- First run may take 10-20 seconds for model training
- Results are cached for faster subsequent runs
- All computations happen locally (no data sent to cloud)
- Model is retrained each time you click "Run Forecast"

---

**Happy Forecasting! 📈**
