import pandas as pd
import itertools
import warnings
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

warnings.filterwarnings("ignore")

def split_data(series: pd.Series, train_ratio: float = 0.8) -> tuple:
    split_idx = int(len(series) * train_ratio)
    return series.iloc[:split_idx], series.iloc[split_idx:]

@st.cache_resource(show_spinner=False)
def find_best_arima(train_series: pd.Series, p_values: range, d_values: range, q_values: range) -> tuple:
    best_aic = float("inf")
    best_order = (1, 1, 1) # Fallback
    
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(train_series, order=(p, d, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, d, q)
        except Exception:
            continue
    return best_order, best_aic

def evaluate_model(train: pd.Series, test: pd.Series, order: tuple):
    model = ARIMA(train, order=order)
    fitted_model = model.fit()
    predictions = fitted_model.forecast(steps=len(test))
    predictions.index = test.index
    return fitted_model, predictions

def forecast_future(series: pd.Series, order: tuple, target_date: str = '2027-06-30') -> pd.DataFrame:
    last_date = series.index[-1]
    target_dt = pd.to_datetime(target_date)
    
    if target_dt <= last_date:
        raise ValueError("Target date must be in the future.")
        
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), end=target_dt)
    
    model = ARIMA(series, order=order)
    fitted_model = model.fit()
    
    forecast_obj = fitted_model.get_forecast(steps=len(future_dates))
    conf_int = forecast_obj.conf_int(alpha=0.05)
    
    return pd.DataFrame({
        'Forecast': forecast_obj.predicted_mean.values,
        'Lower_Bound': conf_int.iloc[:, 0].values,
        'Upper_Bound': conf_int.iloc[:, 1].values
    }, index=future_dates)
