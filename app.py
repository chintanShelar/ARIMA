import streamlit as st
import pandas as pd
from utils.data_loader import load_data, validate_data, fetch_company_profile
from utils.forecasting import split_data, find_best_arima, evaluate_model, forecast_future
from utils.metrics import calculate_metrics
from utils.visualization import plot_historical, plot_actual_vs_predicted, plot_forecast

# Setup UI configuration
st.set_page_config(page_title="Quantitative Forecasting System", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS INJECTION ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    h1, h2, h3 { color: #FFFFFF !important; font-family: 'Inter', sans-serif; }
    
    [data-testid="stMetricValue"] { color: #00E5FF !important; font-size: 2.2rem !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #A0AEC0 !important; font-size: 1rem !important; font-weight: 500; }
    [data-testid="stSidebar"] { background-color: #1A202C; border-right: 1px solid #2D3748; }
    [data-testid="stDataFrame"] { border-radius: 8px; border: 1px solid #2D3748; }
    
    .stButton>button {
        background-color: #00E5FF; color: #000000; font-weight: 600;
        border-radius: 6px; border: none; transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #00B8D4; box-shadow: 0 4px 12px rgba(0, 229, 255, 0.4); }
    hr { border-color: #2D3748 !important; }
    
    /* Custom Corporate Profile Card */
    .profile-card {
        background-color: #1A202C;
        padding: 24px;
        border-radius: 8px;
        border-left: 4px solid #00E5FF;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .profile-title { margin-top: 0; color: #FFFFFF; font-size: 1.8rem; font-weight: 600; margin-bottom: 8px;}
    .profile-subtitle { color: #00E5FF; font-size: 0.95rem; font-weight: 600; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.5px;}
    .profile-summary { color: #A0AEC0; font-size: 1rem; line-height: 1.6; margin-bottom: 0;}
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Institutional Quantitative Forecasting")
st.markdown("<p style='color: #A0AEC0; font-size: 1.1rem;'>Leveraging AutoRegressive Integrated Moving Average (ARIMA) models to project long-term asset valuations.</p>", unsafe_allow_html=True)
st.divider()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Model Parameters")
ticker = st.sidebar.text_input("Target Asset Ticker:", "RELIANCE.NS").upper()
target_date = '2027-06-30'
st.sidebar.markdown(f"**Projection Horizon:** `{target_date}`")

st.sidebar.subheader("ARIMA Optimization Grid")
p_max = st.sidebar.slider("Max AutoRegressive Terms (p)", 1, 5, 2)
d_max = st.sidebar.slider("Max Differencing Order (d)", 0, 2, 1)
q_max = st.sidebar.slider("Max Moving Average Terms (q)", 1, 5, 2)

if st.sidebar.button("Execute Quantitative Analysis", type="primary"):
    if not ticker:
        st.error("Error: Target asset ticker is required for analysis.")
        st.stop()
        
    with st.spinner("Retrieving institutional market data and corporate profile..."):
        df = load_data(ticker)
        profile = fetch_company_profile(ticker)
        
    if not validate_data(df):
        st.error("Error: Insufficient historical data or invalid ticker symbol.")
        st.stop()
        
    # --- CORPORATE OVERVIEW CARD ---
    if profile:
        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-title">{profile.get('name', ticker)}</div>
            <div class="profile-subtitle">{profile.get('sector', 'N/A')} | {profile.get('industry', 'N/A')}</div>
            <div class="profile-summary">{profile.get('summary', '')}</div>
        </div>
        """, unsafe_allow_html=True)
        
    target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    series = df[target_col].dropna()
    
    # --- SECTION 1 ---
    st.header("1. Asset Profile & Historical Context")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Asset Symbol", ticker)
    col2.metric("Latest Valuation", f"₹{series.iloc[-1]:.2f}")
    col3.metric("Data Horizon", f"{series.index[0].date()} to {series.index[-1].date()}")
    col4.metric("Total Observations", len(series))
    
    st.plotly_chart(plot_historical(df, ticker), use_container_width=True)
    
    train, test = split_data(series)
    
    with st.spinner("Executing grid search for optimal autoregressive parameters..."):
        best_order, best_aic = find_best_arima(train, range(0, p_max+1), range(0, d_max+1), range(0, q_max+1))
        
    with st.spinner("Evaluating model efficacy against holdout test data..."):
        fitted_model, predictions = evaluate_model(train, test, best_order)
        metrics = calculate_metrics(test.values, predictions.values)

    st.header("2. Model Architecture & Statistical Validation")
    colA, colB, colC = st.columns(3)
    colA.metric("Optimal ARIMA Vector", str(best_order))
    colB.metric("Mean Absolute Error", f"{metrics['MAE']:.2f}")
    colC.metric("Mean Absolute Percentage Error", f"{metrics['MAPE']:.2f}%")
    
    st.plotly_chart(plot_actual_vs_predicted(train, test, predictions), use_container_width=True)
    
    with st.spinner(f"Projecting valuation trajectories through {target_date}..."):
        forecast_df = forecast_future(series, best_order, target_date)

    st.header(f"3. Strategic Forecast Horizon ({target_date})")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Terminal Projected Valuation", f"₹{forecast_df['Forecast'].iloc[-1]:.2f}")
    fc2.metric("Maximum Projected Bound", f"₹{forecast_df['Forecast'].max():.2f}")
    fc3.metric("Minimum Projected Bound", f"₹{forecast_df['Forecast'].min():.2f}")
    
    st.plotly_chart(plot_forecast(series, forecast_df), use_container_width=True)
