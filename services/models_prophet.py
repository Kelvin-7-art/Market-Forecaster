"""
Prophet forecasting model service for Market Forecaster.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def forecast_prophet(df: pd.DataFrame, horizon: int = 14) -> dict:
    """
    Generate forecast using Facebook Prophet.

    Args:
        df: DataFrame with 'Date' and 'Close' columns
        horizon: Number of days to forecast

    Returns:
        Dictionary with forecast data and metrics
    """
    try:
        from prophet import Prophet
    except Exception as e:
        print(f"Prophet not available: {e}")
        return None
    try:
        prophet_df = df[['Date', 'Close']].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )

        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)

        future_forecast = forecast.tail(horizon)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        future_forecast.columns = ['Date', 'Forecast', 'Lower', 'Upper']

        val_size = min(60, len(df) // 5)
        if val_size > 0:
            train_df = prophet_df.iloc[:-val_size]
            val_df = prophet_df.iloc[-val_size:]

            val_model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            val_model.fit(train_df)

            val_pred = val_model.predict(val_df[['ds']])

            y_true = val_df['y'].values
            y_pred = val_pred['yhat'].values

            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        else:
            rmse, mae, mape = 0, 0, 0

        return {
            'forecast': future_forecast,
            'full_forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            'metrics': {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            },
            'model_name': 'Prophet'
        }

    except Exception as e:
        print(f"Prophet forecast error: {e}")
        return None
