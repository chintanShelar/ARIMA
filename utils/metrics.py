import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    epsilon = np.finfo(np.float64).eps
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
