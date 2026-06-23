import plotly.graph_objects as go
import pandas as pd

# Define our color palette
COLORS = {
    'background': 'rgba(0,0,0,0)', # Transparent to let Streamlit CSS show through
    'primary': '#00E5FF',     # Neon Blue
    'secondary': '#FF007F',   # Neon Pink
    'grid': '#2D3748',        # Subtle dark gray
    'text': '#A0AEC0'         # Muted text
}

def apply_sleek_layout(fig: go.Figure, title: str) -> go.Figure:
    """Helper function to apply consistent dark styling across all charts."""
    fig.update_layout(
        title=dict(text=title, font=dict(color='#FFFFFF', size=20)),
        template='plotly_dark',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        xaxis=dict(showgrid=True, gridcolor=COLORS['grid'], title_font=dict(color=COLORS['text'])),
        yaxis=dict(showgrid=True, gridcolor=COLORS['grid'], title_font=dict(color=COLORS['text'])),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_historical(df: pd.DataFrame, ticker: str) -> go.Figure:
    target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    fig = go.Figure()
    
    # Use a gradient fill under the line for a modern look
    fig.add_trace(go.Scatter(
        x=df.index, y=df[target_col], 
        mode='lines', 
        name=f'{ticker}', 
        line=dict(color=COLORS['primary'], width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 229, 255, 0.1)'
    ))
    
    return apply_sleek_layout(fig, f'{ticker} Market Pulse')

def plot_actual_vs_predicted(train: pd.Series, test: pd.Series, predictions: pd.Series) -> go.Figure:
    fig = go.Figure()
    context_train = train.iloc[-int(len(train)*0.2):]
    
    fig.add_trace(go.Scatter(x=context_train.index, y=context_train, mode='lines', name='Context (Train)', line=dict(color='#4A5568', width=1)))
    fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Actual (Test)', line=dict(color=COLORS['primary'], width=2)))
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions, mode='lines', name='Predicted', line=dict(color=COLORS['secondary'], dash='dot', width=2)))
    
    return apply_sleek_layout(fig, 'Algorithm Validation: Reality vs Prediction')

def plot_forecast(historical: pd.Series, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=historical.index, y=historical, mode='lines', name='Historical', line=dict(color=COLORS['primary'], width=2)))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecast', line=dict(color=COLORS['secondary'], width=3)))
    
    # Styled Confidence Intervals
    fig.add_trace(go.Scatter(
        name='Upper Bound', x=forecast_df.index, y=forecast_df['Upper_Bound'], 
        mode='lines', line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        name='Lower Bound', x=forecast_df.index, y=forecast_df['Lower_Bound'], 
        mode='lines', line=dict(width=0), 
        fillcolor='rgba(255, 0, 127, 0.15)', fill='tonexty', showlegend=False
    ))
    
    return apply_sleek_layout(fig, 'Future Horizon (Extrapolated to June 2027)')
