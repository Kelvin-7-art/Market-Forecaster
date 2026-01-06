"""
Technical indicators calculation service for Market Forecaster.
Computes all technical indicators used in the application.
"""

import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data
def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators for the given OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with all indicators added
    """
    df = df.copy()
    
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df['Volatility_7d'] = df['Returns'].rolling(window=7).std() * np.sqrt(252) * 100
    df['Volatility_14d'] = df['Returns'].rolling(window=14).std() * np.sqrt(252) * 100
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)
    
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute Relative Strength Index (RSI).
    """
    df = df.copy()
    delta = df['Close'].diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    return df


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Compute MACD (Moving Average Convergence Divergence).
    """
    df = df.copy()
    
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    return df


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """
    Compute Bollinger Bands.
    """
    df = df.copy()
    
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    rolling_std = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + (rolling_std * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (rolling_std * std_dev)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
    
    return df


def get_current_indicators(df: pd.DataFrame) -> dict:
    """
    Get the latest values of all indicators.
    """
    if df.empty:
        return {}
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    return {
        'price': latest['Close'],
        'price_change': latest['Close'] - prev['Close'],
        'price_change_pct': ((latest['Close'] - prev['Close']) / prev['Close']) * 100,
        'volume': latest['Volume'],
        'rsi': latest.get('RSI', 50),
        'macd': latest.get('MACD', 0),
        'macd_signal': latest.get('MACD_Signal', 0),
        'sma_20': latest.get('SMA_20', latest['Close']),
        'sma_50': latest.get('SMA_50', latest['Close']),
        'volatility_7d': latest.get('Volatility_7d', 0),
        'volatility_14d': latest.get('Volatility_14d', 0),
        'bb_upper': latest.get('BB_Upper', latest['Close']),
        'bb_lower': latest.get('BB_Lower', latest['Close']),
    }
