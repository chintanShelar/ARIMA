# 📈 Indian Stock Forecasting System using ARIMA

A forward-thinking, end-to-end web application that leverages historical data to predict the future movements of Indian stocks (NSE/BSE) using the ARIMA model. Built completely in Python, driven by Streamlit, and architected for the cloud.

## ✨ Features
- **Live Data Fetching**: Pulls the last 5 years of historical stock data via `yfinance`.
- **Auto ARIMA Logic**: Dynamically grid-searches the optimal (p,d,q) parameters for the lowest AIC.
- **Model Validation**: 80/20 train-test split with MAE, RMSE, and MAPE metrics.
- **Future Forecasting**: Extends the mathematical horizon out to June 30, 2027.
- **Interactive Visuals**: Plotly-powered charts for deep zooming, hovering, and reality checks.
- **Data Export**: One-click CSV downloads for both historical and forecasted datasets.

## 🚀 Local Execution Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/stock-forecast-app.git](https://github.com/your-username/stock-forecast-app.git)
   cd stock-forecast-app
