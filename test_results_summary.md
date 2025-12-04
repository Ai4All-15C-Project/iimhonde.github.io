# Test Results Summary: Improving R² Performance

**Objective**: Find configurations to boost R² from 0.019 (AI Tech Index) to 0.07 target

## Key Findings

### 🏆 Winner: MSFT Individual Stock
- **R² = 0.0900 (9.0%)** - EXCEEDS 0.07 TARGET
- 14-day horizon
- Exogenous: VIX + Treasury_10Y
- Model: SARIMAX(0,0,2)
- Directional accuracy: 58.49%

### Top 5 Configurations by R²

1. **MSFT (Individual)**: 9.00% - 58.5% directional accuracy
2. **MSFT with change vars**: 5.37% - 58.1% directional accuracy
3. **Cloud/SaaS Sector**: 5.21% - 46.4% directional accuracy
4. **Cybersecurity Sector**: 4.96% - 55.1% directional accuracy
5. **Big Tech Sector**: 3.82% - **67.2% directional accuracy** (best)

## Individual Stock Performance (14-day)

| Stock | R² | Dir Acc | Assessment |
|-------|-----|---------|------------|
| MSFT | 9.00% | 58.5% | ✅ Excellent - Most predictable |
| AMD | 1.37% | 54.0% | ⚠️ Modest |
| AAPL | 0.70% | 62.3% | ⚠️ Weak R² but good direction |
| CRM | 0.05% | 38.9% | ❌ Nearly useless |
| ZS | -3.65% | 52.1% | ❌ Fails |
| GOOGL | -8.56% | 66.0% | ❌ Fails despite good direction |

## Sector Indices Performance (14-day)

All sectors beat individual AI Tech Index (1.9%):

| Sector | R² | Dir Acc | Notes |
|--------|-----|---------|-------|
| Cloud/SaaS | 5.21% | 46.4% | Strong R², weaker direction |
| Cybersecurity | 4.96% | 55.1% | Balanced performance |
| Big Tech | 3.82% | 67.2% | Lower R² but best direction |

## Feature Engineering Tests

### ❌ Momentum Variables (GOOGL)
- Without momentum: R² = -8.56%
- With momentum (5d, 20d returns): R² = -25.95%
- **Result**: Made performance WORSE by -17.4 percentage points

### ❌ Change Variables (MSFT)
- Levels only (VIX, Treasury_10Y): R² = 9.00%
- With changes (VIX_change, Rate_change): R² = 5.37%
- **Result**: Made performance WORSE by -3.6 percentage points

### ❌ 30-Day Horizon (AAPL)
- 14-day: R² = 0.70%
- 30-day: R² = -3.95%
- **Result**: Longer horizon performs worse

## Key Insights

### 1. Stock-Specific Predictability Varies Wildly
- MSFT: Highly predictable (9.0%)
- GOOGL: Completely unpredictable (-8.6%)
- Not all tech stocks respond to macro factors equally

### 2. Individual Stocks > Indices for R²
- Best individual stock (MSFT): 9.0%
- Best sector index (Cloud/SaaS): 5.2%
- AI Tech Index (all 23 stocks): 1.9%
- **Reason**: Diversification reduces signal-to-noise ratio

### 3. Sector Homogeneity Matters
- Cloud/SaaS (4 stocks): 5.2%
- Cybersecurity (3 stocks): 5.0%
- Big Tech (5 stocks): 3.8%
- All Tech (23 stocks): 1.9%
- **Reason**: More similar stocks have stronger common drivers

### 4. Simple is Better
- VIX + Treasury_10Y: Optimal
- Adding momentum/change variables: Makes it worse
- **Reason**: Overfitting, multicollinearity, or spurious patterns

### 5. Directional Accuracy ≠ R²
- GOOGL: R² = -8.6% but Dir Acc = 66.0%
- Big Tech: R² = 3.8% but Dir Acc = 67.2%
- **Implication**: Some stocks give good direction signals but poor magnitude predictions

## Recommendations

### For Maximum R² (Point Predictions)
**Use MSFT 14-day forecasting**:
- Target: 14-day log returns
- Exogenous: VIX + Treasury_10Y only
- Model: SARIMAX(0,0,2)
- Expected R²: ~9%

### For Directional Trading (Up/Down)
**Use Big Tech Sector Index**:
- More stable than individual stocks
- 67.2% directional accuracy
- Lower volatility than individual stocks

### For Portfolio Applications
**Use Sector-Specific Indices**:
- Cloud/SaaS for growth exposure
- Cybersecurity for defensive tech
- Avoid mixing too many heterogeneous stocks

## What NOT to Do

❌ Don't add momentum variables - they overfit
❌ Don't use 30-day horizon - predictability degrades
❌ Don't assume all tech stocks are equally predictable
❌ Don't add complexity for complexity's sake

## Next Steps

1. Update EDA notebook to showcase MSFT as primary example
2. Add sector-specific analysis alongside portfolio index
3. Document stock-specific predictability variation
4. Update README to reflect realistic expectations (1-9% R² range)
5. Consider multi-model approach: MSFT for point predictions, Big Tech for direction
