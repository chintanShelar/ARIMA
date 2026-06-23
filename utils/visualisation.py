import plotly.graph_objects as go
import pandas as pd

def plot_historical(df: pd.DataFrame, ticker: str) -> go.Figure:
    target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[target_col], mode='lines', name=f'{ticker} Historical', line=dict(color='#2E86C1')))
    fig.update_layout(title=f'Historical Price Action: {ticker}', template='plotly_white', hovermode='x unified')
    return fig

def plot_actual_vs_predicted(train: pd.Series, test: pd.Series, predictions: pd.Series) -> go.Figure:
    fig = go.Figure()
    context_train = train.iloc[-int(len(train)*0.2):]
    fig.add_trace(go.Scatter(x=context_train.index, y=context_train, mode='lines', name='Train (Context)', line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Actual Test', line=dict(color='#2E86C1')))
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions, mode='lines', name='Predicted Test', line=dict(color='#E74C3C', dash='dash')))
    fig.update_layout(title='Reality Check: Actual vs Predicted', template='plotly_white', hovermode='x unified')
    return fig

def plot_forecast(historical: pd.Series, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical.index, y=historical, mode='lines', name='Historical', line=dict(color='#2E86C1')))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecast', line=dict(color='#E74C3C')))
    fig.add_trace(go.Scatter(name='Upper Bound', x=forecast_df.index, y=forecast_df['Upper_Bound'], mode='lines', marker=dict(color="#444"), line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(name='Lower Bound', x=forecast_df.index, y=forecast_df['Lower_Bound'], marker=dict(color="#444"), line=dict(width=0), mode='lines', fillcolor='rgba(231, 76, 60, 0.2)', fill='tonexty', showlegend=False))
    fig.update_layout(title='Market Horizon: Historical & ARIMA Forecast (June 2027)', template='plotly_white', hovermode='x unified')
    return fig
