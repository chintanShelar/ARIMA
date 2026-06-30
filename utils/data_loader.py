import yfinance as yf
import pandas as pd
import requests
import streamlit as st
from datetime import date
from dateutil.relativedelta import relativedelta

@st.cache_data(ttl=3600)
def search_asset_directory(query: str) -> dict:
    """Hits the shadow API to map human names to market tickers."""
    if not query:
        return {}
        
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    params = {'q': query, 'quotesCount': 6, 'newsCount': 0}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        results = {}
        for q in data.get('quotes', []):
            if 'symbol' in q:
                # Prioritize long name, fallback to short name, fallback to symbol
                name = q.get('longname', q.get('shortname', q['symbol']))
                exchange = q.get('exchDisp', 'Unknown Exchange')
                label = f"{name} ({q['symbol']}) - {exchange}"
                results[label] = q['symbol']
                
        return results
    except Exception as e:
        return {}

@st.cache_data(ttl=86400)
def fetch_company_profile(ticker: str) -> dict:
    """Extracts fundamental corporate data for the stylish overview."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        summary = info.get("longBusinessSummary", "Corporate summary currently unavailable.")
        if len(summary) > 400:
            summary = summary[:397] + "..."
            
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Sector Unknown"),
            "industry": info.get("industry", "Industry Unknown"),
            "summary": summary
        }
    except Exception:
        return {}

@st.cache_data(ttl=86400)
def load_data(ticker: str) -> pd.DataFrame:
    """Fetches the last 5 years of daily stock data."""
    end_date = date.today()
    start_date = end_date - relativedelta(years=5)
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return pd.DataFrame()
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
            
        data = data.asfreq('B')
        data = data.ffill().bfill()
        return data
    except Exception as e:
        st.error(f"System encountered an error during data retrieval: {e}")
        return pd.DataFrame()

def validate_data(data: pd.DataFrame, min_obs: int = 200) -> bool:
    if data is None or data.empty or len(data) < min_obs:
        return False
    if 'Adj Close' not in data.columns and 'Close' not in data.columns:
        return False
    return True
