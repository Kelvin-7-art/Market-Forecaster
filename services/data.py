"""
Data fetching and caching service for Market Forecaster.
Handles downloading OHLCV data from yfinance and data cleaning.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta


@st.cache_data(ttl=300)
def fetch_market_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance.
    
    Args:
        symbol: Ticker symbol (e.g., 'BTC-USD', 'AAPL')
        period: Time period ('1y', '2y', '5y', 'max')
        interval: Data interval ('1d', '1h')
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        elif 'Datetime' in df.columns:
            df['Date'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
            df = df.drop('Datetime', axis=1)
        
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.dropna()
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()


def get_ticker_info(symbol: str) -> dict:
    """
    Get ticker information like name, currency, etc.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'name': info.get('shortName', symbol),
            'currency': info.get('currency', 'USD'),
            'type': info.get('quoteType', 'Unknown'),
            'exchange': info.get('exchange', 'Unknown')
        }
    except:
        return {
            'name': symbol,
            'currency': 'USD',
            'type': 'Unknown',
            'exchange': 'Unknown'
        }


def validate_symbol(symbol: str) -> bool:
    """
    Check if a symbol is valid by attempting to fetch minimal data.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5d")
        return not df.empty
    except:
        return False


def get_popular_symbols() -> dict:
    """
    Return a dictionary of popular symbols by category.
    """
    return {
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD"],
        "Tech Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
        "Indices": ["^SPX", "^IXIC", "^DJI"],
        "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
    }
