import streamlit as st
import pandas as pd
from utils.data_loader import load_data, validate_data
from utils.forecasting import split_data, find_best_arima, evaluate_model, forecast_future
from utils.metrics import calculate_metrics
from utils.visualization import plot_historical, plot_actual_vs_predicted, plot_forecast

# Setup UI configuration - Must be the first Streamlit command
st.set_page_config(page_title="Indian Stock Forecasting", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS INJECTION ---
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metrics container styling */
    [data-testid="stMetricValue"] {
        color: #00E5FF !important; /* Cyber neon blue */
        font-size: 2.2rem !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #A0AEC0 !important;
        font-size: 1rem !important;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1A202C;
        border-right: 1px solid #2D3748;
    }
    
    /* Clean up dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        border: 1px solid #2D3748;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #00E5FF;
        color: #000000;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00B8D4;
        box-shadow: 0 4px 12px rgba(0, 229, 255, 0.4);
    }
    
    /* Dividers */
    hr {
        border-color: #2D3748 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Advanced Market Oracle: ARIMA")
st.markdown("<p style='color: #A0AEC0; font-size: 1.1rem;'>Predicting the market isn't a crystal ball—it's math dancing to the rhythm of past vibes. Horizon: June 2027.</p>", unsafe_allow_html=True)
st.divider()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker:", "RELIANCE.NS").upper()
target_date = '2027-06-30'
st.sidebar.markdown(f"**Target Horizon:** `{target_date}`")

st.sidebar.subheader("ARIMA Parameters")
p_max = st.sidebar.slider("Max AR (p)", 1, 5, 2)
d_max = st.sidebar.slider("Max I (d)", 0, 2, 1)
q_max = st.sidebar.slider("Max MA (q)", 1, 5, 2)

if st.sidebar.button("Generate Forecast", type="primary"):
    if not ticker:
        st.error("Provide a ticker symbol.")
        st.stop()
        
    with st.spinner("Pulling data from the void..."):
        df = load_data(ticker)
        
    if not validate_data(df):
        st.error("Insufficient or invalid data. Check the ticker.")
        st.stop()
        
    target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    series = df[target_col].dropna()
    
    # --- SECTION 1 ---
    st.header("1. Asset Profile")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ticker", ticker)
    col2.metric("Current Price", f"₹{series.iloc[-1]:.2f}")
    col3.metric("Timeline", f"{series.index[0].date()} to {series.index[-1].date()}")
    col4.metric("Observations", len(series))
    
    st.header("2. Historical Price Action")
    st.plotly_chart(plot_historical(df, ticker), use_container_width=True)
    
    train, test = split_data(series)
    
    with st.spinner("Grid searching optimal parameters. Let it cook..."):
        best_order, best_aic = find_best_arima(train, range(0, p_max+1), range(0, d_max+1), range(0, q_max+1))
        
    with st.spinner("Validating against reality..."):
        fitted_model, predictions = evaluate_model(train, test, best_order)
        metrics = calculate_metrics(test.values, predictions.values)

    st.header("3. Architecture & Reality Check")
    colA, colB, colC = st.columns(3)
    colA.metric("ARIMA Logic", str(best_order))
    colB.metric("MAE", f"{metrics['MAE']:.2f}")
    colC.metric("MAPE", f"{metrics['MAPE']:.2f}%")
    
    st.plotly_chart(plot_actual_vs_predicted(train, test, predictions), use_container_width=True)
    
    with st.spinner("Manifesting the timeline to June 2027..."):
        forecast_df = forecast_future(series, best_order, target_date)

    st.header("4. The 2027 Horizon")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Projected Close", f"₹{forecast_df['Forecast'].iloc[-1]:.2f}")
    fc2.metric("Peak Projection", f"₹{forecast_df['Forecast'].max():.2f}")
    fc3.metric("Floor Projection", f"₹{forecast_df['Forecast'].min():.2f}")
    
    st.plotly_chart(plot_forecast(series, forecast_df), use_container_width=True)
