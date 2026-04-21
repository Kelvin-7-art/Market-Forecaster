"""
Market Forecaster - AI Stock & Crypto Price Forecasting Platform
A modern, professional financial analytics dashboard with dark mode UI.
"""

import os
os.environ["STREAMLIT_DATAFRAME_USE_ARROW"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from services.data import fetch_market_data, get_ticker_info, validate_symbol, get_popular_symbols
from services.indicators import compute_all_indicators, get_current_indicators
from services.models_prophet import forecast_prophet
from services.models_arima import forecast_arima
try:
    from services.models_dl import forecast_lstm, forecast_gru
    DL_AVAILABLE = True
except Exception as e:
    DL_AVAILABLE = False
from services.signals import generate_signal, generate_signal_history
from services.backtest import run_backtest, get_buy_and_hold_benchmark

st.set_page_config(
    page_title="Market Forecaster",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

DARK_COLORS = {
    'background': '#0E1117',
    'card_bg': '#1E2130',
    'card_border': '#2D3348',
    'primary': '#00D4AA',
    'primary_gradient': 'linear-gradient(135deg, #00D4AA 0%, #00A3FF 100%)',
    'secondary': '#00A3FF',
    'bullish': '#00E676',
    'bearish': '#FF5252',
    'warning': '#FFB74D',
    'text_primary': '#FFFFFF',
    'text_secondary': '#8B95A5',
    'grid': '#2D3348',
    'plotly_template': 'plotly_dark'
}

LIGHT_COLORS = {
    'background': '#F8FAFC',
    'card_bg': '#FFFFFF',
    'card_border': '#E2E8F0',
    'primary': '#0891B2',
    'primary_gradient': 'linear-gradient(135deg, #0891B2 0%, #0EA5E9 100%)',
    'secondary': '#0EA5E9',
    'bullish': '#10B981',
    'bearish': '#EF4444',
    'warning': '#F59E0B',
    'text_primary': '#1E293B',
    'text_secondary': '#64748B',
    'grid': '#E2E8F0',
    'plotly_template': 'plotly_white'
}

COLORS = LIGHT_COLORS if st.session_state.theme == 'light' else DARK_COLORS

def get_theme_css():
    """Generate CSS based on current theme."""
    is_dark = st.session_state.theme == 'dark'
    c = COLORS
    
    if is_dark:
        bg = '#0E1117'
        card_bg = '#1E2130'
        card_bg_gradient = 'linear-gradient(145deg, #1E2130 0%, #171B26 100%)'
        header_gradient = 'linear-gradient(135deg, #1E2130 0%, #0E1117 100%)'
        sidebar_gradient = 'linear-gradient(180deg, #1E2130 0%, #0E1117 100%)'
    else:
        bg = '#F8FAFC'
        card_bg = '#FFFFFF'
        card_bg_gradient = 'linear-gradient(145deg, #FFFFFF 0%, #F1F5F9 100%)'
        header_gradient = 'linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%)'
        sidebar_gradient = 'linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%)'
    
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    
    .stApp {{
        background-color: {bg};
    }}
    
    .main-header {{
        background: {header_gradient};
        padding: 1.5rem 2rem;
        border-radius: 16px;
        border: 1px solid {c['card_border']};
        margin-bottom: 1.5rem;
    }}
    
    .main-title {{
        font-size: 2rem;
        font-weight: 700;
        background: {c['primary_gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }}
    
    .main-subtitle {{
        color: {c['text_secondary']};
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }}
    
    .metric-card {{
        background: {card_bg_gradient};
        border: 1px solid {c['card_border']};
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, {'0.3' if is_dark else '0.08'});
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        border-color: {c['primary']};
        box-shadow: 0 8px 30px rgba(0, 212, 170, 0.15);
        transform: translateY(-2px);
    }}
    
    .metric-icon {{
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
    }}
    
    .metric-label {{
        color: {c['text_secondary']};
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.35rem;
    }}
    
    .metric-value {{
        color: {c['text_primary']};
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }}
    
    .metric-change {{
        font-size: 0.85rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    
    .bullish {{ color: {c['bullish']}; }}
    .bearish {{ color: {c['bearish']}; }}
    .neutral {{ color: {c['text_secondary']}; }}
    
    .signal-badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1.25rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .signal-buy {{
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.2) 0%, rgba(0, 230, 118, 0.1) 100%);
        color: {c['bullish']};
        border: 1px solid rgba(0, 230, 118, 0.3);
    }}
    
    .signal-sell {{
        background: linear-gradient(135deg, rgba(255, 82, 82, 0.2) 0%, rgba(255, 82, 82, 0.1) 100%);
        color: {c['bearish']};
        border: 1px solid rgba(255, 82, 82, 0.3);
    }}
    
    .signal-hold {{
        background: linear-gradient(135deg, rgba(255, 183, 77, 0.2) 0%, rgba(255, 183, 77, 0.1) 100%);
        color: {c['warning']};
        border: 1px solid rgba(255, 183, 77, 0.3);
    }}
    
    .chart-container {{
        background: {card_bg_gradient};
        border: 1px solid {c['card_border']};
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }}
    
    .section-header {{
        color: {c['text_primary']};
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .info-box {{
        background: rgba(0, 163, 255, 0.1);
        border: 1px solid rgba(0, 163, 255, 0.3);
        border-radius: 12px;
        padding: 1rem;
        color: {c['secondary']};
        font-size: 0.9rem;
    }}
    
    .warning-box {{
        background: rgba(255, 183, 77, 0.1);
        border: 1px solid rgba(255, 183, 77, 0.3);
        border-radius: 12px;
        padding: 1rem;
        color: {c['warning']};
        font-size: 0.9rem;
    }}
    
    .success-box {{
        background: rgba(0, 230, 118, 0.1);
        border: 1px solid rgba(0, 230, 118, 0.3);
        border-radius: 12px;
        padding: 1rem;
        color: {c['bullish']};
        font-size: 0.9rem;
    }}
    
    .danger-box {{
        background: rgba(255, 82, 82, 0.1);
        border: 1px solid rgba(255, 82, 82, 0.3);
        border-radius: 12px;
        padding: 1rem;
        color: {c['bearish']};
        font-size: 0.9rem;
    }}
    
    div[data-testid="stSidebar"] {{
        background: {sidebar_gradient};
        border-right: 1px solid {c['card_border']};
    }}
    
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {c['text_primary']};
    }}
    
    .stButton>button {{
        background: {c['primary_gradient']};
        color: {'#0E1117' if is_dark else '#FFFFFF'};
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        width: 100%;
    }}
    
    .stButton>button:hover {{
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.4);
        transform: translateY(-2px);
    }}
    
    .stSelectbox>div>div, .stTextInput>div>div>input, .stSlider>div>div>div {{
        background-color: {card_bg};
        border: 1px solid {c['card_border']};
        border-radius: 10px;
        color: {c['text_primary']};
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: {card_bg};
        border: 1px solid {c['card_border']};
        border-radius: 10px;
        color: {c['text_secondary']};
        padding: 0.75rem 1.25rem;
        font-weight: 500;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.2) 0%, rgba(0, 163, 255, 0.2) 100%);
        border-color: {c['primary']};
        color: {c['primary']};
    }}
    
    .stDataFrame {{
        background-color: {card_bg};
        border-radius: 12px;
    }}
    
    .stExpander {{
        background-color: {card_bg};
        border: 1px solid {c['card_border']};
        border-radius: 12px;
    }}
    
    div[data-testid="stMetricValue"] {{
        font-size: 1.75rem;
        font-weight: 700;
    }}
    
    .stProgress > div > div > div {{
        background: {c['primary_gradient']};
    }}
    
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {bg};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {c['card_border']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {c['text_secondary']};
    }}
</style>
"""

st.markdown(get_theme_css(), unsafe_allow_html=True)


def safe_display_df(df):
    """Render a dataframe without using pyarrow (Windows DLL safe)."""
    try:
        st.write(df.to_html(index=False), unsafe_allow_html=True)
    except Exception:
        st.write(df)


def create_header():
    """Create the main header section."""
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">Market Forecaster</h1>
        <p class="main-subtitle">AI-Powered Stock & Crypto Price Forecasting Platform</p>
    </div>
    """, unsafe_allow_html=True)


def create_metric_card(icon, label, value, change=None, change_pct=None, is_signal=False, signal_type=None):
    """Create a styled metric card."""
    change_class = ""
    change_html = ""
    
    if change is not None:
        if change > 0:
            change_class = "bullish"
            arrow = "↑"
        elif change < 0:
            change_class = "bearish"
            arrow = "↓"
        else:
            change_class = "neutral"
            arrow = "→"
        
        if change_pct is not None:
            change_html = f'<div class="metric-change {change_class}">{arrow} {abs(change_pct):.2f}%</div>'
    
    if is_signal:
        signal_class = f"signal-{signal_type.lower()}" if signal_type else "signal-hold"
        value_html = f'<span class="signal-badge {signal_class}">{value}</span>'
    else:
        value_html = f'<div class="metric-value">{value}</div>'
    
    icon_bg = "#00D4AA22" if not is_signal else {
        'BUY': '#00E67622',
        'SELL': '#FF525222',
        'HOLD': '#FFB74D22'
    }.get(signal_type, '#00D4AA22')
    
    return f"""
    <div class="metric-card">
        <div class="metric-icon" style="background: {icon_bg};">{icon}</div>
        <div class="metric-label">{label}</div>
        {value_html}
        {change_html}
    </div>
    """


def create_price_chart(df, forecast_data=None, show_volume=True):
    """Create the main price chart with optional forecast overlay."""
    fig = make_subplots(
        rows=2 if show_volume else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.8, 0.2] if show_volume else [1]
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color=COLORS['bullish'],
            decreasing_line_color=COLORS['bearish'],
            increasing_fillcolor=COLORS['bullish'],
            decreasing_fillcolor=COLORS['bearish']
        ),
        row=1, col=1
    )
    
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['SMA_20'],
                name='SMA 20',
                line=dict(color='#FFB74D', width=1.5),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['SMA_50'],
                name='SMA 50',
                line=dict(color='#00A3FF', width=1.5),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    if forecast_data is not None and 'forecast' in forecast_data:
        fc = forecast_data['forecast']
        
        fig.add_trace(
            go.Scatter(
                x=fc['Date'],
                y=fc['Upper'],
                name='Upper Bound',
                line=dict(color='rgba(0, 212, 170, 0.3)', width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=fc['Date'],
                y=fc['Lower'],
                name='Confidence Band',
                fill='tonexty',
                fillcolor='rgba(0, 212, 170, 0.15)',
                line=dict(color='rgba(0, 212, 170, 0.3)', width=0)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=fc['Date'],
                y=fc['Forecast'],
                name='Forecast',
                line=dict(color=COLORS['primary'], width=3, dash='dot')
            ),
            row=1, col=1
        )
    
    if show_volume:
        colors = [COLORS['bullish'] if df['Close'].iloc[i] >= df['Open'].iloc[i] 
                  else COLORS['bearish'] for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        template=COLORS['plotly_template'],
        paper_bgcolor=COLORS['card_bg'],
        plot_bgcolor=COLORS['card_bg'],
        font=dict(family='Inter', color=COLORS['text_primary']),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(30, 33, 48, 0.8)'
        ),
        margin=dict(l=60, r=20, t=40, b=40),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        gridwidth=0.5
    )
    
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        showgrid=True,
        gridwidth=0.5,
        title_text='Price' if show_volume else None,
        row=1, col=1
    )
    
    if show_volume:
        fig.update_yaxes(
            title_text='Volume',
            row=2, col=1
        )
    
    return fig


def create_indicator_chart(df, indicator_type='RSI'):
    """Create indicator sub-charts (RSI, MACD)."""
    fig = go.Figure()
    
    if indicator_type == 'RSI':
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['RSI'],
                name='RSI',
                line=dict(color=COLORS['primary'], width=2)
            )
        )
        
        fig.add_hline(y=70, line_dash='dash', line_color=COLORS['bearish'], opacity=0.7)
        fig.add_hline(y=30, line_dash='dash', line_color=COLORS['bullish'], opacity=0.7)
        fig.add_hrect(y0=30, y1=70, fillcolor='rgba(0, 212, 170, 0.05)', line_width=0)
        
        fig.update_layout(
            title='Relative Strength Index (RSI)',
            yaxis=dict(range=[0, 100])
        )
    
    elif indicator_type == 'MACD':
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['MACD'],
                name='MACD',
                line=dict(color=COLORS['primary'], width=2)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['MACD_Signal'],
                name='Signal',
                line=dict(color=COLORS['warning'], width=2)
            )
        )
        
        colors = [COLORS['bullish'] if val >= 0 else COLORS['bearish'] 
                  for val in df['MACD_Histogram']]
        
        fig.add_trace(
            go.Bar(
                x=df['Date'],
                y=df['MACD_Histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.6
            )
        )
        
        fig.update_layout(title='MACD (12, 26, 9)')
    
    elif indicator_type == 'Bollinger':
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['BB_Upper'],
                name='Upper Band',
                line=dict(color='rgba(0, 163, 255, 0.5)', width=1)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['BB_Lower'],
                name='Lower Band',
                fill='tonexty',
                fillcolor='rgba(0, 163, 255, 0.1)',
                line=dict(color='rgba(0, 163, 255, 0.5)', width=1)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Close'],
                name='Price',
                line=dict(color=COLORS['primary'], width=2)
            )
        )
        
        fig.update_layout(title='Bollinger Bands (20, 2)')
    
    fig.update_layout(
        template=COLORS['plotly_template'],
        paper_bgcolor=COLORS['card_bg'],
        plot_bgcolor=COLORS['card_bg'],
        font=dict(family='Inter', color=COLORS['text_primary']),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=60, r=20, t=60, b=40),
        height=300,
        hovermode='x unified'
    )
    
    fig.update_xaxes(gridcolor=COLORS['grid'], showgrid=True, gridwidth=0.5)
    fig.update_yaxes(gridcolor=COLORS['grid'], showgrid=True, gridwidth=0.5)
    
    return fig


def create_signal_chart(signal_history):
    """Create signal history visualization."""
    if signal_history.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=signal_history['Date'],
            y=signal_history['Close'],
            name='Price',
            line=dict(color=COLORS['text_secondary'], width=1.5),
            opacity=0.7
        )
    )
    
    buy_signals = signal_history[signal_history['Signal'] == 'BUY']
    sell_signals = signal_history[signal_history['Signal'] == 'SELL']
    
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['Date'],
                y=buy_signals['Close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=14,
                    color=COLORS['bullish'],
                    line=dict(color='white', width=1)
                )
            )
        )
    
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['Date'],
                y=sell_signals['Close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(
                    symbol='triangle-down',
                    size=14,
                    color=COLORS['bearish'],
                    line=dict(color='white', width=1)
                )
            )
        )
    
    fig.update_layout(
        template=COLORS['plotly_template'],
        paper_bgcolor=COLORS['card_bg'],
        plot_bgcolor=COLORS['card_bg'],
        font=dict(family='Inter', color=COLORS['text_primary']),
        title='Signal History (Last 90 Days)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=60, r=20, t=60, b=40),
        height=400,
        hovermode='x unified'
    )
    
    fig.update_xaxes(gridcolor=COLORS['grid'], showgrid=True, gridwidth=0.5)
    fig.update_yaxes(gridcolor=COLORS['grid'], showgrid=True, gridwidth=0.5)
    
    return fig


def create_equity_chart(equity_df, benchmark_df=None):
    """Create equity curve chart for backtesting."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=equity_df['Date'],
            y=equity_df['Equity'],
            name='Strategy',
            line=dict(color=COLORS['primary'], width=2.5),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)'
        )
    )
    
    if benchmark_df is not None and not benchmark_df.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_df['Date'],
                y=benchmark_df['Benchmark'],
                name='Buy & Hold',
                line=dict(color=COLORS['text_secondary'], width=2, dash='dash'),
                opacity=0.7
            )
        )
    
    fig.update_layout(
        template=COLORS['plotly_template'],
        paper_bgcolor=COLORS['card_bg'],
        plot_bgcolor=COLORS['card_bg'],
        font=dict(family='Inter', color=COLORS['text_primary']),
        title='Equity Curve',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=60, r=20, t=60, b=40),
        height=400,
        hovermode='x unified'
    )
    
    fig.update_xaxes(gridcolor=COLORS['grid'], showgrid=True, gridwidth=0.5)
    fig.update_yaxes(gridcolor=COLORS['grid'], showgrid=True, gridwidth=0.5, title_text='Portfolio Value ($)')
    
    return fig


def render_sidebar():
    """Render the sidebar controls."""
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <div style="font-weight: 600; color: {COLORS['primary']};">Market Forecaster</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"<span style='color: {COLORS['text_secondary']}; font-size: 0.85rem;'>Theme</span>", unsafe_allow_html=True)
        with col2:
            theme_toggle = st.toggle(
                "Dark" if st.session_state.theme == 'dark' else "Light",
                value=st.session_state.theme == 'light',
                key="theme_toggle",
                help="Toggle between dark and light mode"
            )
            if theme_toggle and st.session_state.theme == 'dark':
                st.session_state.theme = 'light'
                st.rerun()
            elif not theme_toggle and st.session_state.theme == 'light':
                st.session_state.theme = 'dark'
                st.rerun()
        
        st.markdown("---")
        st.markdown("### Asset Selection")
        
        popular = get_popular_symbols()
        category = st.selectbox("Category", list(popular.keys()), index=0)
        
        symbol = st.selectbox(
            "Symbol",
            popular[category],
            index=0
        )
        
        custom_symbol = st.text_input("Or enter custom symbol", placeholder="e.g., NVDA, SOL-USD")
        if custom_symbol:
            symbol = custom_symbol.upper()
        
        st.markdown("### Time Settings")
        
        period = st.selectbox(
            "Historical Period",
            ["1y", "2y", "5y", "max"],
            index=0,
            help="Amount of historical data to analyze"
        )
        
        interval = st.selectbox(
            "Data Interval",
            ["1d", "1h"],
            index=0,
            help="1h data is limited to 730 days"
        )
        
        if interval == "1h":
            st.markdown("""
            <div class="warning-box">
                Hourly data limited to 730 days and may be slower
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Forecast Settings")
        
        horizon = st.slider(
            "Forecast Horizon (days)",
            min_value=7,
            max_value=30,
            value=14,
            help="Number of days to forecast ahead"
        )
        
        model = st.selectbox(
            "Forecasting Model",
            ["Prophet", "ARIMA", "LSTM", "GRU", "Compare All"],
            index=0,
            help="Select the AI model for price forecasting"
        )
        
        st.markdown("### Risk Settings")
        
        vol_threshold = st.slider(
            "Volatility Alert Threshold (%)",
            min_value=1,
            max_value=10,
            value=3,
            help="Alert when daily volatility exceeds this threshold"
        )
        
        st.markdown("---")
        
        run_analysis = st.button("Run Analysis", use_container_width=True)
        
        st.markdown("""
        <div style="margin-top: 2rem; text-align: center; color: #8B95A5; font-size: 0.75rem;">
            <p>Data powered by Yahoo Finance</p>
            <p>© 2024 Market Forecaster</p>
        </div>
        """, unsafe_allow_html=True)
    
    return {
        'symbol': symbol,
        'period': period,
        'interval': interval,
        'horizon': horizon,
        'model': model,
        'vol_threshold': vol_threshold,
        'run_analysis': run_analysis
    }


def main():
    """Main application entry point."""
    create_header()
    
    params = render_sidebar()
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'forecast' not in st.session_state:
        st.session_state.forecast = None
    if 'last_symbol' not in st.session_state:
        st.session_state.last_symbol = None
    
    if params['run_analysis'] or st.session_state.data is None or st.session_state.last_symbol != params['symbol']:
        with st.spinner(f"Fetching data for {params['symbol']}..."):
            df = fetch_market_data(params['symbol'], params['period'], params['interval'])
            
            if df.empty:
                st.error(f"Could not fetch data for symbol: {params['symbol']}. Please check the symbol and try again.")
                return
            
            df = compute_all_indicators(df)
            st.session_state.data = df
            st.session_state.last_symbol = params['symbol']
            st.session_state.forecast = None
    
    df = st.session_state.data
    
    if df is None or df.empty:
        st.info("Select an asset and click 'Run Analysis' to get started.")
        return
    
    indicators = get_current_indicators(df)
    signal_data = generate_signal(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            create_metric_card(
                "", 
                "Current Price",
                f"${indicators['price']:,.2f}",
                indicators['price_change'],
                indicators['price_change_pct']
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        vol = indicators.get('volatility_7d', 0)
        vol_class = "bullish" if vol < 30 else ("warning" if vol < 50 else "bearish")
        st.markdown(
            create_metric_card(
                "",
                "7D Volatility",
                f"{vol:.1f}%"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        rsi = indicators.get('rsi', 50)
        rsi_status = "Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else "Neutral")
        st.markdown(
            create_metric_card(
                "",
                f"RSI ({rsi_status})",
                f"{rsi:.1f}"
            ),
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            create_metric_card(
                "",
                "Trading Signal",
                signal_data['signal'],
                is_signal=True,
                signal_type=signal_data['signal']
            ),
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tabs = st.tabs(["Overview", "Indicators", "Forecast", "Signals", "Backtest", "Alerts"])
    
    with tabs[0]:
        st.markdown('<div class="section-header">Price Chart</div>', unsafe_allow_html=True)
        
        forecast_data = st.session_state.forecast
        fig = create_price_chart(df, forecast_data)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">RSI</div>', unsafe_allow_html=True)
            rsi_fig = create_indicator_chart(df, 'RSI')
            st.plotly_chart(rsi_fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            st.markdown('<div class="section-header">MACD</div>', unsafe_allow_html=True)
            macd_fig = create_indicator_chart(df, 'MACD')
            st.plotly_chart(macd_fig, use_container_width=True, config={'displayModeBar': False})
    
    with tabs[1]:
        st.markdown('<div class="section-header">Technical Indicators</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Moving Averages</div>
                <div style="margin-top: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #8B95A5;">SMA 20:</span>
                        <span style="color: #FFB74D; font-weight: 600;">${:,.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #8B95A5;">SMA 50:</span>
                        <span style="color: #00A3FF; font-weight: 600;">${:,.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #8B95A5;">EMA 20:</span>
                        <span style="color: #00D4AA; font-weight: 600;">${:,.2f}</span>
                    </div>
                </div>
            </div>
            """.format(
                indicators.get('sma_20', 0),
                indicators.get('sma_50', 0),
                df['EMA_20'].iloc[-1] if 'EMA_20' in df.columns else 0
            ), unsafe_allow_html=True)
        
        with col2:
            macd_val = indicators.get('macd', 0)
            macd_sig = indicators.get('macd_signal', 0)
            macd_status = "Bullish" if macd_val > macd_sig else "Bearish"
            macd_color = COLORS['bullish'] if macd_val > macd_sig else COLORS['bearish']
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MACD Analysis</div>
                <div style="margin-top: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #8B95A5;">MACD:</span>
                        <span style="color: #00D4AA; font-weight: 600;">{macd_val:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #8B95A5;">Signal:</span>
                        <span style="color: #FFB74D; font-weight: 600;">{macd_sig:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #8B95A5;">Status:</span>
                        <span style="color: {macd_color}; font-weight: 600;">{macd_status}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            rsi_val = indicators.get('rsi', 50)
            rsi_status = "Oversold" if rsi_val < 30 else ("Overbought" if rsi_val > 70 else "Neutral")
            rsi_color = COLORS['bullish'] if rsi_val < 30 else (COLORS['bearish'] if rsi_val > 70 else COLORS['warning'])
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RSI Analysis</div>
                <div style="margin-top: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #8B95A5;">RSI (14):</span>
                        <span style="color: #00D4AA; font-weight: 600;">{rsi_val:.1f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #8B95A5;">Status:</span>
                        <span style="color: {rsi_color}; font-weight: 600;">{rsi_status}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #8B95A5;">Range:</span>
                        <span style="color: #8B95A5; font-weight: 500;">0 - 100</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        bb_fig = create_indicator_chart(df, 'Bollinger')
        st.plotly_chart(bb_fig, use_container_width=True, config={'displayModeBar': False})
        
        with st.expander("View Raw Indicator Data"):
            display_cols = ['Date', 'Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
            display_df = df[display_cols].tail(30).copy()
            display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
            safe_display_df(display_df)
    
    with tabs[2]:
        st.markdown('<div class="section-header">AI Price Forecast</div>', unsafe_allow_html=True)
        
        if st.button("Generate Forecast", use_container_width=True):
            forecast_results = {}
            
            if params['model'] == "Compare All":
                models_to_run = ["Prophet", "ARIMA", "LSTM", "GRU"]
            else:
                models_to_run = [params['model']]
            
            for model_name in models_to_run:
                try:
                    if model_name == "Prophet":
                        result = forecast_prophet(df, params['horizon'])
                    elif model_name == "ARIMA":
                        result = forecast_arima(df, params['horizon'])
                    elif model_name == "LSTM":
                        if DL_AVAILABLE:
                            result = forecast_lstm(df, params['horizon'])
                        else:
                            st.warning("LSTM model not available (dependency issue)")
                            continue
                    elif model_name == "GRU":
                        if DL_AVAILABLE:
                            result = forecast_gru(df, params['horizon'])
                        else:
                            st.warning("GRU model not available (dependency issue)")
                            continue
                    else:
                        continue

                    if result is not None:
                        forecast_results[model_name] = result

                except Exception as e:
                    st.error(f"{model_name} failed: {str(e)}")

            if len(forecast_results) > 0:
                st.session_state.forecast = list(forecast_results.values())[0]
                
                if params['model'] == "Compare All":
                    st.markdown("### Model Comparison")
                    
                    comparison_data = []
                    for name, result in forecast_results.items():
                        metrics = result['metrics']
                        comparison_data.append({
                            'Model': name,
                            'RMSE': f"${metrics['RMSE']:,.2f}",
                            'MAE': f"${metrics['MAE']:,.2f}",
                            'MAPE': f"{metrics['MAPE']:.2f}%"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    safe_display_df(comparison_df)
                    
                    best_model = min(
                        forecast_results.items(),
                        key=lambda x: x[1]['metrics'].get('RMSE', float('inf'))
                    )
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>Best Model:</strong> {best_model[0]} (lowest RMSE: ${best_model[1]['metrics']['RMSE']:,.2f})
                    </div>
                    """, unsafe_allow_html=True)
        
        if st.session_state.forecast:
            fc = st.session_state.forecast
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = create_price_chart(df.tail(90), fc, show_volume=False)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                st.markdown("### Forecast Details")
                
                forecast_df = fc['forecast'].copy()
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.strftime('%Y-%m-%d')
                forecast_df['Forecast'] = forecast_df['Forecast'].apply(lambda x: f"${x:,.2f}")
                forecast_df['Lower'] = forecast_df['Lower'].apply(lambda x: f"${x:,.2f}")
                forecast_df['Upper'] = forecast_df['Upper'].apply(lambda x: f"${x:,.2f}")
                
                safe_display_df(forecast_df)
                
                metrics = fc['metrics']
                st.markdown(f"""
                <div class="info-box">
                    <strong>Model Performance:</strong><br>
                    RMSE: ${metrics['RMSE']:,.2f}<br>
                    MAE: ${metrics['MAE']:,.2f}<br>
                    MAPE: {metrics['MAPE']:.2f}%
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Click 'Generate Forecast' to run the AI forecasting model.")
    
    with tabs[3]:
        st.markdown('<div class="section-header">Trading Signals</div>', unsafe_allow_html=True)
        
        signal = signal_data['signal']
        strength = signal_data['strength']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            signal_color = {
                'BUY': COLORS['bullish'],
                'SELL': COLORS['bearish'],
                'HOLD': COLORS['warning']
            }.get(signal, COLORS['warning'])
            
            st.markdown(f"""
            <div class="metric-card" style="text-align: center; padding: 2rem;">
                <div class="metric-label">Current Signal</div>
                <div class="signal-badge signal-{signal.lower()}" style="font-size: 1.5rem; margin: 1rem auto;">
                    {signal}
                </div>
                <div class="metric-label" style="margin-top: 1rem;">Signal Strength</div>
                <div class="metric-value">{strength}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Signal Analysis")
            for reason in signal_data['reasons']:
                st.markdown(f"- {reason}")
        
        st.markdown("---")
        
        signal_history = generate_signal_history(df, days=90)
        if not signal_history.empty:
            signal_fig = create_signal_chart(signal_history)
            if signal_fig:
                st.plotly_chart(signal_fig, use_container_width=True, config={'displayModeBar': False})
    
    with tabs[4]:
        st.markdown('<div class="section-header">Strategy Backtesting</div>', unsafe_allow_html=True)
        
        signal_history = generate_signal_history(df, days=len(df))
        
        if not signal_history.empty:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                initial_capital = st.number_input(
                    "Initial Capital ($)",
                    min_value=1000,
                    max_value=1000000,
                    value=10000,
                    step=1000
                )
                
                run_backtest_btn = st.button("Run Backtest", use_container_width=True)
            
            if run_backtest_btn or 'backtest_results' in st.session_state:
                if run_backtest_btn:
                    results = run_backtest(signal_history, initial_capital)
                    benchmark = get_buy_and_hold_benchmark(signal_history, initial_capital)
                    st.session_state.backtest_results = results
                    st.session_state.benchmark = benchmark
                
                results = st.session_state.backtest_results
                benchmark = st.session_state.benchmark
                
                if results['equity_curve'] is not None and not results['equity_curve'].empty:
                    stats = results['stats']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        return_color = "bullish" if stats.get('total_return', 0) > 0 else "bearish"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Total Return</div>
                            <div class="metric-value {return_color}">{stats.get('total_return', 0):.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Max Drawdown</div>
                            <div class="metric-value bearish">{stats.get('max_drawdown', 0):.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Win Rate</div>
                            <div class="metric-value">{stats.get('win_rate', 0):.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        sharpe_color = "bullish" if stats.get('sharpe_ratio', 0) > 1 else ("warning" if stats.get('sharpe_ratio', 0) > 0 else "bearish")
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Sharpe Ratio</div>
                            <div class="metric-value {sharpe_color}">{stats.get('sharpe_ratio', 0):.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    equity_fig = create_equity_chart(results['equity_curve'], benchmark)
                    st.plotly_chart(equity_fig, use_container_width=True, config={'displayModeBar': False})
                    
                    with st.expander("Trade Log"):
                        if results['trades']:
                            trades_df = pd.DataFrame(results['trades'])
                            trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
                            trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:,.2f}")
                            if 'pnl' in trades_df.columns:
                                trades_df['pnl'] = trades_df['pnl'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")
                            if 'pnl_pct' in trades_df.columns:
                                trades_df['pnl_pct'] = trades_df['pnl_pct'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
                            safe_display_df(trades_df)
                        else:
                            st.info("No trades executed during the backtest period.")
        else:
            st.warning("Insufficient data for backtesting. Please ensure you have enough historical data.")
    
    with tabs[5]:
        st.markdown('<div class="section-header">Volatility Alerts & Risk Analysis</div>', unsafe_allow_html=True)
        
        vol_7d = indicators.get('volatility_7d', 0)
        vol_14d = indicators.get('volatility_14d', 0)
        
        if vol_7d > 50:
            risk_level = "HIGH"
            risk_color = COLORS['bearish']
            risk_bg = "rgba(255, 82, 82, 0.2)"
        elif vol_7d > 30:
            risk_level = "MEDIUM"
            risk_color = COLORS['warning']
            risk_bg = "rgba(255, 183, 77, 0.2)"
        else:
            risk_level = "LOW"
            risk_color = COLORS['bullish']
            risk_bg = "rgba(0, 230, 118, 0.2)"
        
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; margin-bottom: 1.5rem;">
            <div class="metric-label">Current Risk Status</div>
            <div style="display: inline-block; padding: 0.75rem 2rem; border-radius: 50px; 
                        background: {risk_bg}; border: 1px solid {risk_color}; margin-top: 1rem;">
                <span style="color: {risk_color}; font-weight: 700; font-size: 1.5rem;">{risk_level} RISK</span>
            </div>
            <div style="margin-top: 1rem; color: #8B95A5;">
                7-Day Volatility: <strong style="color: {risk_color};">{vol_7d:.1f}%</strong> | 
                14-Day Volatility: <strong>{vol_14d:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        vol_threshold = params['vol_threshold']
        
        if 'Volatility_7d' in df.columns:
            high_vol_days = df[df['Volatility_7d'] > vol_threshold * 10].tail(10)
            
            if not high_vol_days.empty:
                st.markdown(f"""
                <div class="danger-box">
                    <strong>{len(high_vol_days)} high volatility events</strong> detected in recent history 
                    (threshold: {vol_threshold}% daily change)
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("High Volatility Days"):
                    display_df = high_vol_days[['Date', 'Close', 'Volatility_7d']].copy()
                    display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
                    display_df['Volatility_7d'] = display_df['Volatility_7d'].apply(lambda x: f"{x:.1f}%")
                    display_df.columns = ['Date', 'Close Price', '7D Volatility']
                    safe_display_df(display_df)
            else:
                st.markdown("""
                <div class="success-box">
                    No high volatility events detected based on current threshold.
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("### Volatility Chart")
        
        vol_fig = go.Figure()
        
        vol_fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Volatility_7d'],
                name='7-Day Volatility',
                line=dict(color=COLORS['primary'], width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 170, 0.1)'
            )
        )
        
        vol_fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Volatility_14d'],
                name='14-Day Volatility',
                line=dict(color=COLORS['secondary'], width=2)
            )
        )
        
        vol_fig.add_hline(
            y=vol_threshold * 10,
            line_dash='dash',
            line_color=COLORS['bearish'],
            annotation_text=f"Alert Threshold ({vol_threshold}%)",
            annotation_position="right"
        )
        
        vol_fig.update_layout(
            template=COLORS['plotly_template'],
            paper_bgcolor=COLORS['card_bg'],
            plot_bgcolor=COLORS['card_bg'],
            font=dict(family='Inter', color=COLORS['text_primary']),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=20, t=40, b=40),
            height=350,
            hovermode='x unified',
            yaxis_title='Annualized Volatility (%)'
        )
        
        vol_fig.update_xaxes(gridcolor=COLORS['grid'], showgrid=True, gridwidth=0.5)
        vol_fig.update_yaxes(gridcolor=COLORS['grid'], showgrid=True, gridwidth=0.5)
        
        st.plotly_chart(vol_fig, use_container_width=True, config={'displayModeBar': False})


if __name__ == "__main__":
    main()
