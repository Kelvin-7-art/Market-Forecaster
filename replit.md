# Market Forecaster - AI Stock & Crypto Price Forecasting Platform

## Overview
Market Forecaster is a modern, professional financial analytics dashboard built with Streamlit. It provides AI-powered price forecasting for stocks and cryptocurrencies using multiple machine learning models.

## Features
- **Real-time Market Data**: Fetches OHLCV data from Yahoo Finance
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
- **AI Forecasting Models**: Prophet, ARIMA, LSTM, GRU with model comparison
- **Trading Signals**: Rule-based Buy/Hold/Sell signals with explanations
- **Backtesting**: Strategy backtesting with equity curves and performance metrics
- **Volatility Alerts**: Risk monitoring with customizable thresholds
- **Theme Toggle**: Dark mode (default) with optional light mode

## Project Structure
```
├── app.py                    # Main Streamlit application
├── services/
│   ├── __init__.py
│   ├── data.py              # Data fetching and caching
│   ├── indicators.py        # Technical indicators calculation
│   ├── models_prophet.py    # Prophet forecasting model
│   ├── models_arima.py      # ARIMA forecasting model
│   ├── models_dl.py         # LSTM/GRU forecasting models
│   ├── signals.py           # Trading signal generation
│   └── backtest.py          # Backtesting logic
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── pyproject.toml           # Python dependencies
```

## UI Design
- **Style**: Clean fintech aesthetic inspired by Bloomberg/TradingView
- **Color Palette**: 
  - Dark mode: Deep navy background (#0E1117), teal/blue accents
  - Light mode: Light gray background (#F8FAFC), cyan accents
- **Typography**: Inter font family with clear hierarchy
- **Layout**: Left sidebar for controls, main content with cards and charts

## Tabs
1. **Overview**: Price chart with candlesticks, RSI, MACD indicators
2. **Indicators**: Detailed technical indicator analysis and Bollinger Bands
3. **Forecast**: AI model predictions with confidence intervals
4. **Signals**: Current trading signal with history visualization
5. **Backtest**: Strategy performance testing and equity curves
6. **Alerts**: Volatility monitoring and risk status

## Running the Application
The application runs automatically via the workflow:
```bash
streamlit run app.py --server.port 5000
```

## Dependencies
- streamlit, pandas, numpy, plotly
- yfinance (market data)
- prophet, statsmodels (forecasting)
- scikit-learn (metrics and scaling)

## User Preferences
- Default symbol: BTC-USD
- Default period: 1 year
- Default theme: Dark mode
