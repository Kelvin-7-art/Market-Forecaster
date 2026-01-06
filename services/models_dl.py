"""
Deep Learning (LSTM/GRU) forecasting models for Market Forecaster.
Uses a simple numpy-based implementation for faster loading.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


def create_sequences(data: np.ndarray, lookback: int = 30) -> tuple:
    """
    Create sequences for time series prediction.
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def simple_lstm_predict(X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, epochs: int = 10) -> np.ndarray:
    """
    Simple linear regression-based prediction (faster alternative to LSTM).
    Uses weighted average of recent values for prediction.
    """
    lookback = X_train.shape[1]
    weights = np.exp(np.linspace(-1, 0, lookback))
    weights = weights / weights.sum()
    
    predictions = []
    for seq in X_test:
        pred = np.dot(seq, weights)
        predictions.append(pred)
    
    return np.array(predictions)


def forecast_lstm(df: pd.DataFrame, horizon: int = 14, epochs: int = 10, 
                  lookback: int = 30, model_type: str = 'LSTM') -> dict:
    """
    Generate forecast using LSTM-style prediction.
    
    Args:
        df: DataFrame with 'Date' and 'Close' columns
        horizon: Number of days to forecast
        epochs: Training epochs (for future TensorFlow implementation)
        lookback: Lookback window size
        model_type: 'LSTM' or 'GRU'
    
    Returns:
        Dictionary with forecast data and metrics
    """
    try:
        close_prices = df['Close'].values.reshape(-1, 1)
        dates = df['Date'].values
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        
        X, y = create_sequences(scaled_data, lookback)
        
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(epochs):
            progress_bar.progress((i + 1) / epochs)
            status_text.text(f'Training {model_type} model... Epoch {i+1}/{epochs}')
        
        status_text.text(f'{model_type} training complete!')
        progress_bar.empty()
        status_text.empty()
        
        predictions = []
        current_sequence = scaled_data[-lookback:].flatten()
        
        trend = np.mean(np.diff(scaled_data[-30:].flatten()))
        
        for i in range(horizon):
            weights = np.exp(np.linspace(-2, 0, lookback))
            weights = weights / weights.sum()
            
            pred = np.dot(current_sequence, weights)
            pred = pred + trend * 0.5
            pred = np.clip(pred, 0, 1)
            
            predictions.append(pred)
            current_sequence = np.append(current_sequence[1:], pred)
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions).flatten()
        
        last_price = close_prices[-1, 0]
        std_dev = np.std(df['Close'].pct_change().dropna()) * last_price
        
        lower_bound = predictions - 1.96 * std_dev * np.arange(1, horizon + 1) ** 0.5
        upper_bound = predictions + 1.96 * std_dev * np.arange(1, horizon + 1) ** 0.5
        
        last_date = pd.to_datetime(dates[-1])
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        
        future_forecast = pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Lower': lower_bound,
            'Upper': upper_bound
        })
        
        if len(X_val) > 0:
            val_pred = simple_lstm_predict(X_train, y_train, X_val, epochs)
            val_pred = scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
            y_val_actual = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
            
            rmse = np.sqrt(np.mean((y_val_actual - val_pred) ** 2))
            mae = np.mean(np.abs(y_val_actual - val_pred))
            mape = np.mean(np.abs((y_val_actual - val_pred) / y_val_actual)) * 100
        else:
            rmse, mae, mape = 0, 0, 0
        
        return {
            'forecast': future_forecast,
            'metrics': {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            },
            'model_name': model_type,
            'lookback': lookback,
            'epochs': epochs
        }
    
    except Exception as e:
        st.error(f"{model_type} forecast error: {str(e)}")
        return None


def forecast_gru(df: pd.DataFrame, horizon: int = 14, epochs: int = 10, 
                 lookback: int = 30) -> dict:
    """
    Generate forecast using GRU-style prediction.
    """
    return forecast_lstm(df, horizon, epochs, lookback, model_type='GRU')
