# Python Financial Analysis Script Explanation

This document provides a comprehensive explanation of the Python script designed for financial analysis and trading signal generation for a list of Indian stock tickers.

## 1. Initial Setup and Import Statements

The script begins with encoding declaration and standard library imports:

```python
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 16:56:31 2025

@author: ashwinik
"""

import yfinance as yf
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
```

*   **`# -*- coding: utf-8 -*-`**: This is a character encoding declaration, specifying that the source code file is encoded using UTF-8. UTF-8 supports a wide range of characters, making the script more portable across different systems and locales. In Python 3, UTF-8 is the default, but explicitly declaring it is good practice.
*   **Docstring**: The multiline comment is a module-level docstring, providing metadata such as the creation date and author. This is useful for tracking and collaboration.
*   **`import yfinance as yf`**: Imports the `yfinance` library, aliased as `yf`. This library is used to download historical market data from Yahoo Finance (e.g., stock prices, volumes).
*   **`import pandas as pd`**: Imports the `pandas` library, aliased as `pd`. Pandas is essential for data manipulation and analysis, providing data structures like DataFrames (for tabular data) and Series.
*   **`import numpy as np`**: Imports the `NumPy` library, aliased as `np`. NumPy is the fundamental package for numerical computation in Python, providing support for arrays and mathematical functions.
*   **`from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer`**: Imports the `SentimentIntensityAnalyzer` class from the `vaderSentiment` library. VADER (Valence Aware Dictionary and sEntiment Reasoner) is a tool specifically designed for sentiment analysis of text, especially social media content.
*   **`from sklearn.preprocessing import MinMaxScaler`**: Imports the `MinMaxScaler` class from the `scikit-learn` library's `preprocessing` module. This class is used to scale numerical features to a specific range, typically 0 to 1, which can be important for certain machine learning algorithms or data analysis tasks.

## 2. Technical Indicator Calculation Functions

The script defines three functions to calculate common technical indicators:

### `calculate_rsi(df, window=14)`
```python
def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```
*   **Purpose**: Calculates the Relative Strength Index (RSI), a momentum oscillator measuring the speed and change of price movements. RSI ranges from 0 to 100.
*   **Parameters**:
    *   `df`: Pandas DataFrame with a 'Close' column for closing prices.
    *   `window` (default `14`): Lookback period for calculating average gains and losses.
*   **Calculation**:
    1.  Calculates price differences (`delta`).
    2.  Separates gains and calculates their rolling mean.
    3.  Separates losses (as positive values) and calculates their rolling mean.
    4.  Computes Relative Strength (`rs` = average gain / average loss).
    5.  Normalizes RS to the 0-100 RSI scale.
*   **Returns**: A pandas Series with RSI values.

### `calculate_macd(df, short_window=12, long_window=26, signal_window=9)`
```python
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd - signal
```
*   **Purpose**: Calculates the Moving Average Convergence Divergence (MACD) histogram. MACD is a trend-following momentum indicator showing the relationship between two Exponential Moving Averages (EMAs).
*   **Parameters**:
    *   `df`: Pandas DataFrame with a 'Close' column.
    *   `short_window` (default `12`): Period for the shorter-term EMA.
    *   `long_window` (default `26`): Period for the longer-term EMA.
    *   `signal_window` (default `9`): Period for the EMA of the MACD line (to create the signal line).
*   **Calculation**:
    1.  Calculates the short-term EMA and long-term EMA of closing prices.
    2.  Calculates the MACD line (`short_ema - long_ema`).
    3.  Calculates the signal line (EMA of the MACD line).
    4.  Returns the MACD histogram (`macd - signal`).
*   **Returns**: A pandas Series with MACD histogram values.

### `calculate_atr(df, window=14)`
```python
def calculate_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()
```
*   **Purpose**: Calculates the Average True Range (ATR), a measure of market volatility.
*   **Parameters**:
    *   `df`: Pandas DataFrame with 'High', 'Low', and 'Close' columns.
    *   `window` (default `14`): Lookback period for averaging the True Range.
*   **Calculation**:
    1.  Calculates three ranges: current High - current Low, absolute (current High - previous Close), and absolute (current Low - previous Close).
    2.  The True Range (TR) for each period is the maximum of these three ranges.
    3.  Calculates the ATR as the rolling mean of the TR values.
*   **Returns**: A pandas Series with ATR values.

## 3. List of Stock Tickers

```python
nse_50_tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
    "ITC.NS", "ASIANPAINT.NS", "HCLTECH.NS", "MARUTI.NS", "LT.NS", "AXISBANK.NS",
    "DMART.NS", "BAJAJFINSV.NS", "WIPRO.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "NESTLEIND.NS", "ONGC.NS", "SUNPHARMA.NS", "JSWSTEEL.NS", "POWERGRID.NS",
    "ADANIPORTS.NS", "TATAMOTORS.NS", "NTPC.NS", "BAJAJ-AUTO.NS"
    ]
```
*   **`nse_50_tickers`**: This is a Python list containing 30 stock ticker symbols.
*   **Stock Exchange**: These tickers represent companies listed on the National Stock Exchange of India (NSE).
*   **`.NS` Suffix**: The `.NS` suffix is used by Yahoo Finance (and thus `yfinance`) to specifically identify securities listed on the NSE. This ensures that data fetched using these tickers corresponds to their NSE listings.
*   **Purpose**: This list defines the specific set of Indian stocks that the script will analyze. The main execution block will iterate through these tickers.

## 4. Agent Classes

The script defines several "agent" classes, each responsible for a specific part of the analysis.

### `WarrenBuffettAgent`
```python
class WarrenBuffettAgent:
    def analyze(self, ticker):
        stock = yf.Ticker(ticker)
        financials = stock.financials
        def safe_get(field, default=0):
            # Check if financials is not empty and field exists
            if not financials.empty and field in financials.index:
                # Attempt to get the first available value (most recent year)
                # financials.loc[field] can be a Series, so we take the first element if available
                value = financials.loc[field].iloc[0] if not financials.loc[field].empty else default
                return value
            return default
            
        net_income = safe_get('Net Income', np.nan)
        equity = safe_get('Total Stockholder Equity', np.nan)
        total_debt = safe_get('Total Debt', np.nan) # Changed from 'Total Debt' to 'Total Liabilities Net Minority Interest' if 'Total Debt' is not available
        if np.isnan(total_debt): # Fallback if 'Total Debt' is not found or results in NaN
             total_debt = safe_get('Total Liabilities Net Minority Interest', np.nan)

        revenue = safe_get('Total Revenue', np.nan)
        operating_income = safe_get('Operating Income', np.nan)
        
        ROE = net_income / (equity + 1e-6) if pd.notna(net_income) and pd.notna(equity) and equity != 0 else 0.15
        Debt_to_Equity = total_debt / (equity + 1e-6) if pd.notna(total_debt) and pd.notna(equity) and equity != 0 else 0.5
        OPM = operating_income / (revenue + 1e-6) if pd.notna(operating_income) and pd.notna(revenue) and revenue != 0 else 0.25
        
        ratios = {'ROE': ROE, 'Debt-to-Equity': Debt_to_Equity, 'OPM': OPM}
        # print("Financial Ratios:", ratios) # Optional: can be verbose
        return ratios
```
*   **Purpose**: Performs a simplified fundamental analysis inspired by value investing principles, focusing on profitability, debt, and operational efficiency.
*   **`analyze(self, ticker)` method**:
    *   Fetches financial statements for the given `ticker` using `yf.Ticker(ticker).financials`.
    *   Uses an inner helper function `safe_get` to retrieve financial data points (Net Income, Total Stockholder Equity, Total Debt, Total Revenue, Operating Income) safely, providing `np.nan` if data is missing or the financials table is empty. It attempts to use the first column's data (most recent year). A fallback for 'Total Debt' to 'Total Liabilities Net Minority Interest' is added.
    *   Calculates three key financial ratios:
        *   **Return on Equity (ROE)**: `net_income / equity`. Defaults to `0.15` if data is missing or equity is zero.
        *   **Debt-to-Equity**: `total_debt / equity`. Defaults to `0.5` if data is missing or equity is zero.
        *   **Operating Profit Margin (OPM)**: `operating_income / revenue`. Defaults to `0.25` if data is missing or revenue is zero.
    *   A small value `1e-6` is added to denominators to prevent division by zero and handles cases where equity or revenue might be zero.
    *   Returns a dictionary of these calculated (or defaulted) ratios.

### `SentimentAgent`
```python
class SentimentAgent:
    def analyze(self, news_articles):
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(article)['compound'] for article in news_articles]
        return np.mean(scores) if scores else 0 # Return 0 if no articles
```
*   **Purpose**: Designed to analyze a collection of news articles and quantify their overall sentiment.
*   **`analyze(self, news_articles)` method**:
    *   Takes a list of `news_articles` (strings) as input.
    *   Uses `SentimentIntensityAnalyzer` from the `vaderSentiment` library.
    *   For each article, it calculates polarity scores and extracts the `compound` score (a single measure of sentiment ranging from -1 to +1).
    *   Returns the mean (average) of these compound scores. If `news_articles` is empty, it returns 0.
*   **Note**: While defined, this agent is **not actively used** in the main execution block of the script as provided.

### `TechnicalAgent`
```python
class TechnicalAgent:
    def analyze(self, df):
        df['RSI'] = calculate_rsi(df) / 100
        df['MACD'] = calculate_macd(df)
        df['ATR'] = calculate_atr(df)
        # print("Technical Indicators (Means):", df[['RSI', 'MACD', 'ATR']].mean().to_dict()) # Optional
        return df[['RSI', 'MACD', 'ATR']].copy() # Return a copy
```
*   **Purpose**: Calculates technical indicators and adds them to the stock data DataFrame.
*   **`analyze(self, df)` method**:
    *   Takes a pandas DataFrame `df` (containing historical stock prices) as input.
    *   Calls the previously defined `calculate_rsi`, `calculate_macd`, and `calculate_atr` functions.
    *   Adds new columns to the input `df` for 'RSI', 'MACD', and 'ATR'.
        *   RSI is normalized by dividing by 100 (scaling it to a 0-1 range).
    *   The input DataFrame `df` is modified in place.
    *   Returns a *copy* of the DataFrame subset containing only the 'RSI', 'MACD', and 'ATR' columns.

### `RiskManager`
```python
class RiskManager:
    def evaluate(self, agent_signals):
        # print("Agent Signals for Risk Evaluation:", agent_signals) # Optional
        weights = {
            'ROE': 0.3, 'Debt-to-Equity': -0.1, # Negative weight: higher D/E increases risk score
            'OPM': 0.3, 
            ('RSI', ''): 0.15, # Assuming RSI here is a score where higher is better (e.g. not overbought)
            ('MACD', ''): 0.1, 
            ('ATR', ''): -0.05 # Negative weight: higher ATR (volatility) increases risk score
        }
        
        risk_score = 0
        # ROE, OPM: Higher is better (contributes positively if signal is high)
        # Debt-to-Equity: Lower is better. If D/E is high, its product with negative weight will be very negative.
        # This suggests risk_score should be interpreted as: higher score = lower risk.
        # Let's adjust logic: a higher score means higher risk.
        # ROE: Positive weight. Higher ROE -> higher score. (Good)
        # OPM: Positive weight. Higher OPM -> higher score. (Good)
        # Debt-to-Equity: Positive weight. Higher D/E -> higher score. (Bad)
        # ATR: Positive weight. Higher ATR -> higher score. (Bad)
        # RSI: If RSI is normalized 0-1 (0-100), usually mid-range is good.
        #      Overbought (e.g. >0.7) is risky. Oversold (e.g. <0.3) can also be risky or opportunity.
        #      Let's assume the input RSI for risk is such that a higher value is "better" for this calculation, or the weight adjusts.
        #      Given the override condition, the raw RSI (0-1) is used.
        # MACD: Positive MACD is generally bullish.

        # Recalibrating weights interpretation: Higher score = higher risk
        weights_recalibrated = {
            'ROE': -0.3,  # Higher ROE is good, so it REDUCES risk score
            'Debt-to-Equity': 0.2, # Higher D/E is bad, INCREASES risk score
            'OPM': -0.3,   # Higher OPM is good, REDUCES risk score
            ('RSI', ''): 0.1, # Assuming RSI is raw (0-1). If RSI > 0.7 (overbought), it adds to risk.
                              # This weight might need to be conditional or RSI transformed.
                              # For simplicity, let's say a high RSI value (e.g. 0.7-1.0) adds to risk.
            ('MACD', ''): -0.05, # Positive MACD is good, REDUCES risk score
            ('ATR', ''): 0.15  # Higher ATR (volatility) is bad, INCREASES risk score
        }

        risk_score = 0
        for key_spec, weight in weights_recalibrated.items():
            signal_value = agent_signals.get(key_spec, 0) # Handles string or tuple keys
            if key_spec == ('RSI', ''): 
                # If RSI > 0.7 (overbought), consider it as contributing to risk directly
                # If RSI < 0.3 (oversold), it might also be risky, or an opportunity
                # Let's make a simple assumption: risk increases if RSI is > 0.7 or < 0.3
                if signal_value > 0.7 or signal_value < 0.3:
                    risk_score += abs(signal_value - 0.5) * weight # Penalize deviation from midpoint
                else: # RSI in mid-range 0.3-0.7 is considered less risky for this component
                    risk_score -= 0.05 # Bonus for being in a good range
            elif key_spec == 'Debt-to-Equity':
                 risk_score += signal_value * weight # Direct: higher D/E increases score
            elif key_spec == ('ATR', ''):
                 risk_score += signal_value * weight # Direct: higher ATR increases score
            else: # For ROE, OPM, MACD where higher is better (negative weight)
                risk_score += signal_value * weight
        
        # Normalize risk score conceptually - this example doesn't have a fixed scale.
        # The threshold 0.35 was for a different scoring logic.
        # Let's set a new threshold based on this additive logic.
        # Example: if score > 0 (positive accumulated risk factors outweigh negative ones) -> High risk
        risk_level = "High" if risk_score > 0.05 else "Low" # Adjusted threshold

        # Risk override (original logic)
        # This override implies that specific technical conditions can make it "Low" risk
        # regardless of the calculated score.
        rsi_val_override = agent_signals.get(('RSI', ''), 0)
        macd_val_override = agent_signals.get(('MACD', ''), 0)
        if 0.5 <= rsi_val_override <= 0.6 and macd_val_override > -0.05:
            risk_level = "Low"
            
        return risk_level
```
*   **Purpose**: Assesses overall investment risk by consolidating signals from fundamental and technical analysis.
*   **`evaluate(self, agent_signals)` method**:
    *   Takes an `agent_signals` dictionary as input.
    *   Defines `weights_recalibrated`. The interpretation is: a higher final `risk_score` means higher risk.
        *   Positive weights for detrimental signals (e.g., high Debt-to-Equity, high ATR).
        *   Negative weights for beneficial signals (e.g., high ROE, high OPM, positive MACD).
    *   RSI contribution is nuanced: risk increases if RSI is in overbought (>0.7) or oversold (<0.3) regions, penalized by deviation from midpoint (0.5); otherwise, a small bonus is given.
    *   Calculates `risk_score` by summing weighted signal values. `agent_signals.get(key_spec, 0)` handles missing signals.
    *   Determines `risk_level`: "High" if `risk_score > 0.05` (adjusted threshold), else "Low".
    *   Includes a "Risk override": if normalized RSI (from `agent_signals`) is between 0.5-0.6 and MACD > -0.05, `risk_level` is set to "Low".
    *   Returns the final `risk_level` string.

### `PortfolioManager`
```python
class PortfolioManager:
    def decide_action(self, risk, rsi, macd):
        if risk == "Low" and 0.4 <= rsi <= 0.6 and macd > -0.05: # Allow slight MACD negativity
            return "Buy"
        elif risk == "Low" and rsi < 0.4: # Low risk but RSI is weak/oversold
            return "Hold"
        elif risk == "High" and rsi > 0.7: # High risk and RSI is overbought
            return "Sell"
        # Additional Hold condition: Low risk but RSI is high (e.g. overbought but not high risk overall)
        elif risk == "Low" and rsi > 0.6:
            return "Hold" 
        # Default Hold for other High risk scenarios or unhandled conditions
        return "Hold"
```
*   **Purpose**: Makes a trading decision ("Buy", "Sell", "Hold") based on risk level and specific technical indicators.
*   **`decide_action(self, risk, rsi, macd)` method**:
    *   **Parameters**:
        *   `risk` (string): The assessed risk level ("Low" or "High").
        *   `rsi` (float): The current RSI value (normalized 0-1).
        *   `macd` (float): The current MACD value.
    *   **Decision Logic**:
        *   **Buy**: If `risk` is "Low", RSI is between 0.4-0.6 (neutral), and MACD > -0.05.
        *   **Hold (Condition 1)**: If `risk` is "Low" and RSI < 0.4 (weak or oversold).
        *   **Sell**: If `risk` is "High" and RSI > 0.7 (overbought).
        *   **Hold (Condition 2)**: If `risk` is "Low" and RSI > 0.6 (overbought but overall risk is low).
        *   **Hold (default)**: For all other conditions (e.g., most "High" risk scenarios not meeting the "Sell" criteria).
    *   Returns the decision as a string: "Buy", "Sell", or "Hold".

## 5. Main Execution Block (`if __name__ == "__main__":`)

This block orchestrates the analysis for each ticker in `nse_50_tickers`.

```python
if __name__ == "__main__":
    results = []
    for ticker in nse_50_tickers:
        try:
            print(f"\nProcessing {ticker}...")
            warren_agent = WarrenBuffettAgent()
            technical_agent = TechnicalAgent()
            risk_manager = RiskManager()
            portfolio_manager = PortfolioManager()

            financial_ratios = warren_agent.analyze(ticker)
            stock_data = yf.download(ticker, period='1y', interval='1d', progress=False) # progress=False to reduce console noise
            
            if stock_data.empty:
                print(f"No data downloaded for {ticker}, skipping.")
                results.append([ticker, "Error - No Data", "N/A"])
                continue
            
            # Ensure 'Close', 'High', 'Low' are present for technical_agent
            required_cols = ['Close', 'High', 'Low']
            if not all(col in stock_data.columns for col in required_cols):
                print(f"Missing required columns for {ticker} (e.g., {', '.join(required_cols)}), skipping.")
                results.append([ticker, f"Error - Missing Columns ({', '.join(required_cols)})", "N/A"])
                continue

            tech_signals_df = technical_agent.analyze(stock_data.copy()) 

            mean_tech_signals = tech_signals_df.mean().to_dict()

            combined_signals_for_risk = {**financial_ratios}
            for k, v in mean_tech_signals.items():
                combined_signals_for_risk[(k, '')] = v 

            risk_level = risk_manager.evaluate(combined_signals_for_risk)
            
            # For PortfolioManager, use the most recent RSI and MACD if available, else mean.
            # If using daily data for 1 year, the last row of tech_signals_df has the latest values.
            if not tech_signals_df.empty:
                latest_rsi = tech_signals_df['RSI'].iloc[-1] if pd.notna(tech_signals_df['RSI'].iloc[-1]) else mean_tech_signals.get('RSI', 0)
                latest_macd = tech_signals_df['MACD'].iloc[-1] if pd.notna(tech_signals_df['MACD'].iloc[-1]) else mean_tech_signals.get('MACD', 0)
            else: # Should not happen if stock_data was not empty and analyze worked
                latest_rsi = mean_tech_signals.get('RSI', 0)
                latest_macd = mean_tech_signals.get('MACD', 0)

            action = portfolio_manager.decide_action(risk_level, latest_rsi, latest_macd)
            results.append([ticker, risk_level, action])
            print(f"Finished processing {ticker}: Risk={risk_level}, Action={action}")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            results.append([ticker, "Error processing", str(e)]) 

    results_df = pd.DataFrame(results, columns=["Stock", "Risk Level", "Trading Decision"])
    print("\n--- Final Results ---")
    print(results_df.to_string()) # .to_string() to print full DataFrame
```

*   **Initialization**: An empty list `results` is created to store the outcomes.
*   **Loop**: Iterates through each `ticker` in `nse_50_tickers`.
    *   **Agent Instantiation**: New instances of agents are created.
    *   **Fundamental Analysis**: `financial_ratios = warren_agent.analyze(ticker)`.
    *   **Data Download**: `stock_data = yf.download(...)` fetches 1 year of daily data. `progress=False` is added.
    *   **Data Check**: Skips if `stock_data` is empty or if required columns ('Close', 'High', 'Low') are missing.
    *   **Technical Analysis**: `tech_signals_df = technical_agent.analyze(stock_data.copy())`.
    *   **Signal Combination for RiskManager**:
        *   Mean of `tech_signals_df` columns (`mean_tech_signals`).
        *   `financial_ratios` combined with `mean_tech_signals`. Keys for technical signals are converted to tuples (e.g., `('RSI', '')`) for `RiskManager`.
    *   **Risk Evaluation**: `risk_level = risk_manager.evaluate(combined_signals_for_risk)`.
    *   **Decision Making for PortfolioManager**:
        *   Uses the *most recent* RSI and MACD values from `tech_signals_df.iloc[-1]`. If the latest value is NaN, it falls back to the mean value for that indicator.
        *   `action = portfolio_manager.decide_action(risk_level, latest_rsi, latest_macd)`.
    *   **Store Results**: Appends `ticker`, `risk_level`, `action` to `results`.
*   **Error Handling**: `try-except` catches errors, prints them, and records them in `results`.
*   **Final Output**:
    *   `results` list converted to `results_df` DataFrame.
    *   Printed to console using `results_df.to_string()` for better display of all rows.

## 6. Overall Script Logic and Flow

1.  The script defines tools: technical indicator functions and agent classes for fundamental analysis, technical analysis, risk management, and portfolio decisions.
2.  It iterates through `nse_50_tickers`.
3.  For each ticker:
    a.  Fetches fundamental ratios using `WarrenBuffettAgent`.
    b.  Downloads 1 year of daily stock prices.
    c.  Calculates time-series technical indicators (RSI, MACD, ATR) using `TechnicalAgent`.
    d.  `RiskManager` evaluates risk using fundamental ratios and *mean* values of technical indicators.
    e.  `PortfolioManager` makes a trading decision ("Buy", "Sell", "Hold") based on this risk level and the *most recent* RSI and MACD values (with a fallback to mean if recent values are not available).
4.  Results are compiled into a DataFrame and printed.

The script employs a modular strategy, combining fundamental data with both mean-based (for risk assessment) and recent-based (for trading decisions) technical analysis to generate actionable insights.
