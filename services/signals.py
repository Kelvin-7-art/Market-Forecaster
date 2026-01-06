"""
Trading signals generation service for Market Forecaster.
Generates Buy/Hold/Sell signals based on technical indicators.
"""

import pandas as pd
import numpy as np


def generate_signal(df: pd.DataFrame) -> dict:
    """
    Generate trading signal based on technical indicators.
    
    Rules:
    - BUY if price above SMA50 AND RSI < 70 AND MACD > signal line
    - SELL if price below SMA50 OR RSI > 75 OR MACD < signal line
    - Else HOLD
    
    Args:
        df: DataFrame with indicators computed
    
    Returns:
        Dictionary with signal and explanation
    """
    if df.empty or len(df) < 50:
        return {
            'signal': 'HOLD',
            'strength': 50,
            'reasons': ['Insufficient data for analysis'],
            'details': {}
        }
    
    latest = df.iloc[-1]
    
    price = latest['Close']
    sma50 = latest.get('SMA_50', price)
    rsi = latest.get('RSI', 50)
    macd = latest.get('MACD', 0)
    macd_signal = latest.get('MACD_Signal', 0)
    
    reasons = []
    buy_score = 0
    sell_score = 0
    
    if price > sma50:
        buy_score += 1
        reasons.append(f"Price (${price:,.2f}) is above SMA50 (${sma50:,.2f}) - Bullish trend")
    else:
        sell_score += 1
        reasons.append(f"Price (${price:,.2f}) is below SMA50 (${sma50:,.2f}) - Bearish trend")
    
    if rsi < 30:
        buy_score += 2
        reasons.append(f"RSI ({rsi:.1f}) indicates oversold conditions - Strong buy signal")
    elif rsi < 70:
        buy_score += 1
        reasons.append(f"RSI ({rsi:.1f}) is in neutral zone - No extreme readings")
    elif rsi > 75:
        sell_score += 2
        reasons.append(f"RSI ({rsi:.1f}) indicates overbought conditions - Strong sell signal")
    else:
        sell_score += 1
        reasons.append(f"RSI ({rsi:.1f}) is approaching overbought territory")
    
    if macd > macd_signal:
        buy_score += 1
        reasons.append(f"MACD ({macd:.4f}) is above signal line ({macd_signal:.4f}) - Bullish momentum")
    else:
        sell_score += 1
        reasons.append(f"MACD ({macd:.4f}) is below signal line ({macd_signal:.4f}) - Bearish momentum")
    
    total_score = buy_score - sell_score
    
    if total_score >= 2:
        signal = 'BUY'
        strength = min(100, 60 + (total_score * 10))
    elif total_score <= -2:
        signal = 'SELL'
        strength = min(100, 60 + (abs(total_score) * 10))
    else:
        signal = 'HOLD'
        strength = 50
    
    return {
        'signal': signal,
        'strength': strength,
        'reasons': reasons,
        'details': {
            'price': price,
            'sma50': sma50,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'buy_score': buy_score,
            'sell_score': sell_score
        }
    }


def generate_signal_history(df: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    """
    Generate signal history for the last N days.
    
    Args:
        df: DataFrame with indicators computed
        days: Number of days to look back
    
    Returns:
        DataFrame with signal history
    """
    if df.empty or len(df) < 50:
        return pd.DataFrame()
    
    history_df = df.tail(days).copy()
    
    signals = []
    for idx in range(len(history_df)):
        row = history_df.iloc[idx]
        
        price = row['Close']
        sma50 = row.get('SMA_50', price)
        rsi = row.get('RSI', 50)
        macd = row.get('MACD', 0)
        macd_signal = row.get('MACD_Signal', 0)
        
        buy_score = 0
        sell_score = 0
        
        if price > sma50:
            buy_score += 1
        else:
            sell_score += 1
        
        if rsi < 30:
            buy_score += 2
        elif rsi < 70:
            buy_score += 1
        elif rsi > 75:
            sell_score += 2
        else:
            sell_score += 1
        
        if macd > macd_signal:
            buy_score += 1
        else:
            sell_score += 1
        
        total = buy_score - sell_score
        
        if total >= 2:
            signals.append('BUY')
        elif total <= -2:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    
    history_df['Signal'] = signals
    
    return history_df[['Date', 'Close', 'Signal', 'RSI', 'MACD', 'MACD_Signal', 'SMA_50']]
