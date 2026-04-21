"""
ARIMA forecasting model service for Market Forecaster.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def forecast_arima(df: pd.DataFrame, horizon: int = 14) -> dict:
    """
    Generate forecast using ARIMA model.

    Args:
        df: DataFrame with 'Date' and 'Close' columns
        horizon: Number of days to forecast

    Returns:
        Dictionary with forecast data and metrics
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as e:
        print(f"ARIMA not available: {e}")
        return None
    try:
        close_prices = df['Close'].values
        dates = df['Date'].values

        best_aic = float('inf')
        best_order = (1, 1, 1)

        orders_to_try = [
            (1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2),
            (1, 0, 1), (2, 0, 1), (0, 1, 1), (1, 1, 0)
        ]

        for order in orders_to_try:
            try:
                model = ARIMA(close_prices, order=order)
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = order
            except Exception:
                continue

        model = ARIMA(close_prices, order=best_order)
        fitted = model.fit()

        forecast_result = fitted.get_forecast(steps=horizon)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)

        last_date = pd.to_datetime(dates[-1])
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')

        future_forecast = pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast_mean,
            'Lower': conf_int[:, 0],
            'Upper': conf_int[:, 1]
        })

        val_size = min(60, len(close_prices) // 5)
        if val_size > 0:
            train_data = close_prices[:-val_size]
            val_data = close_prices[-val_size:]

            val_model = ARIMA(train_data, order=best_order)
            val_fitted = val_model.fit()
            val_pred = val_fitted.forecast(steps=val_size)

            rmse = np.sqrt(np.mean((val_data - val_pred) ** 2))
            mae = np.mean(np.abs(val_data - val_pred))
            mape = np.mean(np.abs((val_data - val_pred) / val_data)) * 100
        else:
            rmse, mae, mape = 0, 0, 0

        return {
            'forecast': future_forecast,
            'metrics': {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            },
            'model_name': f'ARIMA{best_order}',
            'order': best_order
        }

    except Exception as e:
        print(f"ARIMA forecast error: {e}")
        return None
