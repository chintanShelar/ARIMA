import streamlit as st
import pandas as pd
from datetime import datetime

# Local module imports
from utils.data_loader import load_data, validate_data
from utils.forecasting import split_data, find_best_arima, evaluate_model, forecast_future
from utils.metrics import calculate_metrics
from utils.visualization import plot_historical, plot_actual_vs_predicted, plot_forecast

# Setup UI configuration
st.set_page_config(page_title="Indian Stock Forecasting System", layout="wide")

st.title("📈 Indian Stock Forecasting System using ARIMA")
st.markdown("Predicting the market isn't a crystal ball—it's math dancing to the rhythm of past vibes. Let's look forward to June 2027.")

# --- SIDEBAR ---
st.sidebar.header("Configuration Panel")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, TCS.NS):", "RELIANCE.NS").upper()
st.sidebar.markdown("**Forecast End Date:** Fixed to June 30, 2027")
target_date = '2027-06-30'

st.sidebar.subheader("ARIMA Search Grid Space")
p_max = st.sidebar.slider("Max AR (p)", 1, 5, 3)
d_max = st.sidebar.slider("Max I (d)", 0, 2, 1)
q_max = st.sidebar.slider("Max MA (q)", 1, 5, 3)

run_forecast = st.sidebar.button("Generate Forecast", type="primary")

if run_forecast:
    if not ticker:
        st.error("No cap, you need to provide a ticker symbol.")
        st.stop()
        
    with st.spinner("Summoning the ghosts of past data..."):
        df = load_data(ticker)
        
    if not validate_data(df):
        st.error("Insufficient or invalid data. Make sure it's a valid Yahoo Finance ticker (like INFY.NS).")
        st.stop()
        
    target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    series = df[target_col].dropna()
    
    # --- SECTION 1: Stock Information ---
    st.header("1. Stock Information")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ticker Symbol", ticker)
    col2.metric("Current Price (Last Close)", f"₹{series.iloc[-1]:.2f}")
    col3.metric("Data Range", f"{series.index[0].strftime('%Y-%m-%d')} to {series.index[-1].strftime('%Y-%m-%d')}")
    col4.metric("Total Observations", len(series))
    st.divider()

    # --- SECTION 2: Historical Data Table ---
    st.header("2. Historical Data Table")
    st.dataframe(df.sort_index(ascending=False), use_container_width=True, height=250)
    st.divider()

    # --- SECTION 3: Historical Price Chart ---
    st.header("3. Historical Price Chart")
    st.plotly_chart(plot_historical(df, ticker), use_container_width=True)
    st.divider()

    # --- MODELING: Train/Test Split & Grid Search ---
    train, test = split_data(series)
    
    with st.spinner("Grid searching optimal ARIMA parameters. Let the math cook..."):
        best_order, best_aic = find_best_arima(train, range(0, p_max+1), range(0, d_max+1), range(0, q_max+1))
        
    with st.spinner("Validating model against test data..."):
        fitted_model, predictions = evaluate_model(train, test, best_order)
        metrics = calculate_metrics(test.values, predictions.values)

    # --- SECTION 4: ARIMA Model Information & Validation ---
    st.header("4. Model Architecture & Validation")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Optimal ARIMA Order", f"({best_order[0]}, {best_order[1]}, {best_order[2]})")
    colB.metric("AIC Score", f"{best_aic:.2f}")
    colC.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:.2f}")
    colD.metric("MAPE", f"{metrics['MAPE']:.2f}%")
    
    with st.expander("Show Detailed Model Summary"):
        st.text(fitted_model.summary().as_text())
        
    st.plotly_chart(plot_actual_vs_predicted(train, test, predictions), use_container_width=True)
    st.divider()

    # --- FORECASTING ---
    with st.spinner("Extending the timeline to June 2027..."):
        try:
            forecast_df = forecast_future(series, best_order, target_date)
        except Exception as e:
            st.error(f"Forecast generation failed. Check the algorithmic matrix: {e}")
            st.stop()

    # --- SECTION 5: Forecast Results ---
    st.header(f"5. Forecast Horizon (Through {target_date})")
    
    final_price = forecast_df['Forecast'].iloc[-1]
    high_price = forecast_df['Forecast'].max()
    low_price = forecast_df['Forecast'].min()
    avg_price = forecast_df['Forecast'].mean()
    
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Predicted Close (June 30, 2027)", f"₹{final_price:.2f}")
    fc2.metric("Highest Forecast", f"₹{high_price:.2f}")
    fc3.metric("Lowest Forecast", f"₹{low_price:.2f}")
    fc4.metric("Average Forecast", f"₹{avg_price:.2f}")
    
    st.dataframe(forecast_df, use_container_width=True, height=250)
    st.divider()

    # --- SECTION 6: Forecast Graph ---
    st.header("6. Visualizing the Future")
    st.plotly_chart(plot_forecast(series, forecast_df), use_container_width=True)
    st.divider()

    # --- SECTION 7: Download Options ---
    st.header("7. Extract Artifacts")
    dl_col1, dl_col2 = st.columns(2)
    
    @st.cache_data
    def convert_df(df_to_convert):
        return df_to_convert.to_csv().encode('utf-8')
        
    hist_csv = convert_df(df)
    forecast_csv = convert_df(forecast_df)
    
    dl_col1.download_button(
        label="Download Historical Data (CSV)",
        data=hist_csv,
        file_name=f"{ticker}_historical.csv",
        mime="text/csv",
    )
    
    dl_col2.download_button(
        label="Download Forecast Data (CSV)",
        data=forecast_csv,
        file_name=f"{ticker}_forecast_2027.csv",
        mime="text/csv",
    )
