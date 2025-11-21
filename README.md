# 15C_Company-Success-Research-

##Factors to Note:
1. Trading doesn't take place on the weekends, hence the lack of data on the weekends
2. Some of the Companies IPO'd after the pandemic

1. KEY CREATIVE VARIABLES TO PRIORITIZE:
   - Yield_Curve_Slope: Leading recession indicator
   - Vol_of_Vol_Ratio: Market uncertainty measure
   - Semi_vs_Tech_Ratio: AI hardware strength
   - Crypto_Tech_Corr_20d: Risk sentiment
   - Small_vs_Large_Caps: Market breadth
   - Gold_Oil_Ratio: Macro sentiment

2. REGIME-BASED MODELING:
   Use regime indicators to build separate models or weight predictions:
   - AI_Boom_Period for post-ChatGPT dynamics
   - Fed_Hike_Period for rate sensitivity
   - Tech_Bear_2022 for drawdown patterns

3. FEATURE ENGINEERING IDEAS:
   - Interaction terms: VIX × Fed_Hike_Period
   - Rolling correlations between stocks
   - Lagged features (t-1, t-5, t-20)
   - Technical indicators (RSI, MACD)