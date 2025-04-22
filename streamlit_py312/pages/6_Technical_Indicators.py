import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.stock_info import display_stock_info, get_stock_info
from utils.alpaca_api import get_historical_data

# Technical indicators calculation functions
def calculate_sma(data, period=20):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_ema(data, period=20):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # For RSI calculation after first period
    avg_gain = avg_gain.fillna(gain.iloc[:period].mean())
    avg_loss = avg_loss.fillna(loss.iloc[:period].mean())
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(data_high, data_low, data_close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = data_low.rolling(window=k_period).min()
    highest_high = data_high.rolling(window=k_period).max()
    
    k = 100 * ((data_close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_atr(data_high, data_low, data_close, period=14):
    """Calculate Average True Range"""
    high_low = data_high - data_low
    high_close = abs(data_high - data_close.shift())
    low_close = abs(data_low - data_close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_adx(data_high, data_low, data_close, period=14):
    """Calculate Average Directional Index"""
    # Calculate True Range
    high_low = data_high - data_low
    high_close = abs(data_high - data_close.shift())
    low_close = abs(data_low - data_close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    # Calculate Directional Movement
    up_move = data_high - data_high.shift()
    down_move = data_low.shift() - data_low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
    
    # Calculate ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=period).mean()
    
    return adx, plus_di, minus_di

def calculate_obv(data_close, data_volume):
    """Calculate On-Balance Volume"""
    obv = pd.Series(index=data_close.index)
    obv.iloc[0] = 0
    
    for i in range(1, len(data_close)):
        if data_close.iloc[i] > data_close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + data_volume.iloc[i]
        elif data_close.iloc[i] < data_close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - data_volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_ichimoku(data_high, data_low, data_close, tenkan_period=9, kijun_period=26, senkou_b_period=52):
    """Calculate Ichimoku Cloud"""
    # Tenkan-sen (Conversion Line)
    tenkan_sen = (data_high.rolling(window=tenkan_period).max() + data_low.rolling(window=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line)
    kijun_sen = (data_high.rolling(window=kijun_period).max() + data_low.rolling(window=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B)
    senkou_span_b = ((data_high.rolling(window=senkou_b_period).max() + data_low.rolling(window=senkou_b_period).min()) / 2).shift(kijun_period)
    
    # Chikou Span (Lagging Span)
    chikou_span = data_close.shift(-kijun_period)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def calculate_vwap(data_high, data_low, data_close, data_volume):
    """Calculate Volume Weighted Average Price (VWAP)"""
    typical_price = (data_high + data_low + data_close) / 3
    vwap = (typical_price * data_volume).cumsum() / data_volume.cumsum()
    return vwap

def calculate_pivot_points(data_high, data_low, data_close):
    """Calculate Pivot Points (Standard)"""
    pivot_point = (data_high + data_low + data_close) / 3
    support1 = (2 * pivot_point) - data_high
    resistance1 = (2 * pivot_point) - data_low
    support2 = pivot_point - (data_high - data_low)
    resistance2 = pivot_point + (data_high - data_low)
    support3 = data_low - 2 * (data_high - pivot_point)
    resistance3 = data_high + 2 * (pivot_point - data_low)
    
    return pivot_point, support1, resistance1, support2, resistance2, support3, resistance3

def calculate_keltner_channel(data_high, data_low, data_close, ema_period=20, atr_period=10, multiplier=2):
    """Calculate Keltner Channel"""
    ema = calculate_ema(data_close, ema_period)
    atr = calculate_atr(data_high, data_low, data_close, atr_period)
    
    upper = ema + (multiplier * atr)
    lower = ema - (multiplier * atr)
    
    return upper, ema, lower

def calculate_fibonacci_retracement(data_high, data_low):
    """Calculate Fibonacci Retracement Levels"""
    max_price = data_high.max()
    min_price = data_low.min()
    diff = max_price - min_price
    
    level_0 = max_price
    level_23_6 = max_price - (0.236 * diff)
    level_38_2 = max_price - (0.382 * diff)
    level_50_0 = max_price - (0.5 * diff)
    level_61_8 = max_price - (0.618 * diff)
    level_100 = min_price
    
    return level_0, level_23_6, level_38_2, level_50_0, level_61_8, level_100

def calculate_mfi(data_high, data_low, data_close, data_volume, period=14):
    """Calculate Money Flow Index"""
    typical_price = (data_high + data_low + data_close) / 3
    raw_money_flow = typical_price * data_volume
    
    positive_flow = pd.Series(index=typical_price.index)
    negative_flow = pd.Series(index=typical_price.index)
    
    positive_flow.iloc[0] = 0
    negative_flow.iloc[0] = 0
    
    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.iloc[i] = raw_money_flow.iloc[i]
            negative_flow.iloc[i] = 0
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            positive_flow.iloc[i] = 0
            negative_flow.iloc[i] = raw_money_flow.iloc[i]
        else:
            positive_flow.iloc[i] = 0
            negative_flow.iloc[i] = 0
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfr = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + mfr))
    
    return mfi

def calculate_cci(data_high, data_low, data_close, period=20):
    """Calculate Commodity Channel Index"""
    typical_price = (data_high + data_low + data_close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = abs(typical_price - sma_tp).rolling(window=period).mean()
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    
    return cci

def calculate_williams_r(data_high, data_low, data_close, period=14):
    """Calculate Williams %R"""
    highest_high = data_high.rolling(window=period).max()
    lowest_low = data_low.rolling(window=period).min()
    
    williams_r = -100 * (highest_high - data_close) / (highest_high - lowest_low)
    
    return williams_r

def calculate_rate_of_change(data, period=9):
    """Calculate Rate of Change (ROC)"""
    roc = ((data / data.shift(period)) - 1) * 100
    return roc

def get_market_data_for_indicators(ticker='AAPL', days=120):
    """Get real market data for technical indicators using Alpaca API"""
    try:
        # Check if we're in a Streamlit session with credentials stored
        api_key = None
        api_secret = None
        
        if 'alpaca_api_key' in st.session_state and 'alpaca_api_secret' in st.session_state:
            api_key = st.session_state['alpaca_api_key']
            api_secret = st.session_state['alpaca_api_secret']
        # Try to get from Streamlit secrets if available
        elif hasattr(st, 'secrets') and 'alpaca' in st.secrets:
            api_key = st.secrets['alpaca']['api_key']
            api_secret = st.secrets['alpaca']['api_secret']
        # Try to get from environment variables
        else:
            api_key = os.environ.get('ALPACA_API_KEY')
            api_secret = os.environ.get('ALPACA_API_SECRET')
        
        if not api_key or not api_secret:
            st.warning("Alpaca API credentials not found. Using mock data for now.")
            return generate_mock_data(ticker, days)
        
        # Get historical data from Alpaca API
        # Calculate timeframe based on days
        timeframe = '1D'  # Default to daily
        if days <= 7:
            timeframe = '1H'  # If less than a week, use hourly data
        
        # Get historical data
        df = get_historical_data(
            api_key=api_key,
            api_secret=api_secret,
            symbol=ticker,
            timeframe=timeframe,
            period_days=days,
            paper=True  # Use paper trading API
        )
        
        if df.empty:
            st.warning(f"No data available for {ticker}. Using mock data for now.")
            return generate_mock_data(ticker, days)
        
        # Reset index to get the date as a column
        df = df.reset_index()
        
        # Rename columns to match our expected format
        df = df.rename(columns={
            'timestamp': 'Date',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'open': 'Open'
        })
        
        # Drop any rows with NaN values
        df = df.dropna()
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return generate_mock_data(ticker, days)
        
def generate_mock_data(ticker='AAPL', days=120):
    """Generate mock price data for indicators when real data is unavailable"""
    # This function remains as a fallback but will be logged
    st.warning("Using mock data for indicators. This is not recommended for live trading.")
    
    np.random.seed(42)  # For reproducibility
    
    # Start with a base price
    if ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
        base_price = 150.0
    elif ticker in ['TSLA', 'NVDA']:
        base_price = 200.0
    else:
        base_price = 100.0
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price series with a slight upward trend and volatility
    close_prices = [base_price]
    high_prices = [base_price * 1.02]
    low_prices = [base_price * 0.98]
    volumes = [int(np.random.normal(1000000, 200000))]
    
    for i in range(1, len(date_range)):
        # Add random walk with slight upward bias for close price
        close_change = np.random.normal(0.0005, 0.015)  # Mean slightly positive
        new_close = close_prices[-1] * (1 + close_change)
        
        # High and low based on close with some randomness
        daily_volatility = np.random.uniform(0.005, 0.025)
        new_high = new_close * (1 + daily_volatility)
        new_low = new_close * (1 - daily_volatility)
        
        # Ensure high > close > low
        if new_high < new_close:
            new_high = new_close * 1.005
        if new_low > new_close:
            new_low = new_close * 0.995
        
        # Generate volume with occasional spikes
        new_volume = int(np.random.normal(1000000, 200000))
        if np.random.random() < 0.05:  # 5% chance of a volume spike
            new_volume *= np.random.uniform(1.5, 3.0)
        
        close_prices.append(new_close)
        high_prices.append(new_high)
        low_prices.append(new_low)
        volumes.append(new_volume)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': date_range,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    })
    
    return df

def plot_price_chart(df, ticker):
    """Plot main price chart with volume"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                         vertical_spacing=0.03, 
                         row_heights=[0.7, 0.3])
    
    # Add price candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'] if 'Open' in df.columns else df['Close'].shift(1),  # Use actual open if available
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume bar chart
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name="Volume",
            marker_color='rgba(0, 0, 255, 0.3)'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Price Chart',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def plot_moving_averages(df, ticker):
    """Plot price with multiple moving averages"""
    # Calculate moving averages
    df['SMA_20'] = calculate_sma(df['Close'], 20)
    df['SMA_50'] = calculate_sma(df['Close'], 50)
    df['SMA_200'] = calculate_sma(df['Close'], 200)
    df['EMA_10'] = calculate_ema(df['Close'], 10)
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Price',
            line=dict(color='black', width=1)
        )
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='blue', width=1.5)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='green', width=1.5)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['SMA_200'],
            mode='lines',
            name='SMA 200',
            line=dict(color='red', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['EMA_10'],
            mode='lines',
            name='EMA 10',
            line=dict(color='purple', width=1, dash='dash')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['EMA_20'],
            mode='lines',
            name='EMA 20',
            line=dict(color='orange', width=1, dash='dash')
        )
    )
    
    fig.update_layout(
        title=f'{ticker} Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_bollinger_bands(df, ticker):
    """Plot Bollinger Bands"""
    upper, middle, lower = calculate_bollinger_bands(df['Close'], period=20, std_dev=2)
    
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Price',
            line=dict(color='black', width=1)
        )
    )
    
    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=upper,
            mode='lines',
            name='Upper BB',
            line=dict(color='red', width=1)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=middle,
            mode='lines',
            name='SMA 20',
            line=dict(color='blue', width=1)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=lower,
            mode='lines',
            name='Lower BB',
            line=dict(color='green', width=1)
        )
    )
    
    # Add fill between upper and lower bands
    fig.add_trace(
        go.Scatter(
            x=df['Date'].tolist() + df['Date'].tolist()[::-1],
            y=upper.tolist() + lower.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0, 176, 246, 0.1)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='BB Range',
            showlegend=False
        )
    )
    
    fig.update_layout(
        title=f'{ticker} Bollinger Bands (20, 2)',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_rsi(df, ticker):
    """Plot RSI with overbought and oversold levels"""
    rsi = calculate_rsi(df['Close'], period=14)
    
    fig = go.Figure()
    
    # Add RSI line
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=rsi,
            mode='lines',
            name='RSI (14)',
            line=dict(color='blue', width=1.5)
        )
    )
    
    # Add overbought and oversold lines
    fig.add_shape(
        type="line",
        x0=df['Date'].iloc[0],
        y0=70,
        x1=df['Date'].iloc[-1],
        y1=70,
        line=dict(color="red", width=1, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=df['Date'].iloc[0],
        y0=30,
        x1=df['Date'].iloc[-1],
        y1=30,
        line=dict(color="green", width=1, dash="dash"),
    )
    
    # Add midline
    fig.add_shape(
        type="line",
        x0=df['Date'].iloc[0],
        y0=50,
        x1=df['Date'].iloc[-1],
        y1=50,
        line=dict(color="gray", width=1, dash="dash"),
    )
    
    # Add labels for the lines
    fig.add_annotation(
        x=df['Date'].iloc[-1],
        y=70,
        text="Overbought (70)",
        showarrow=False,
        yshift=10,
        font=dict(size=10, color="red")
    )
    
    fig.add_annotation(
        x=df['Date'].iloc[-1],
        y=30,
        text="Oversold (30)",
        showarrow=False,
        yshift=-10,
        font=dict(size=10, color="green")
    )
    
    fig.update_layout(
        title=f'{ticker} Relative Strength Index (RSI)',
        xaxis_title='Date',
        yaxis_title='RSI',
        yaxis=dict(range=[0, 100]),
        height=300,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def plot_macd(df, ticker):
    """Plot MACD (Moving Average Convergence Divergence)"""
    macd_line, signal_line, histogram = calculate_macd(df['Close'])
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                         vertical_spacing=0.03, 
                         row_heights=[0.7, 0.3])
    
    # Add price to top plot
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Price',
            line=dict(color='black', width=1)
        ),
        row=1, col=1
    )
    
    # Add MACD and Signal lines
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=macd_line,
            mode='lines',
            name='MACD Line',
            line=dict(color='blue', width=1.5)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=signal_line,
            mode='lines',
            name='Signal Line',
            line=dict(color='red', width=1.5)
        ),
        row=2, col=1
    )
    
    # Add histogram
    colors = ['green' if val > 0 else 'red' for val in histogram]
    
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=histogram,
            name='Histogram',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Add zero line for the MACD panel
    fig.add_shape(
        type="line",
        x0=df['Date'].iloc[0],
        y0=0,
        x1=df['Date'].iloc[-1],
        y1=0,
        line=dict(color="gray", width=1, dash="dash"),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{ticker} MACD (12, 26, 9)',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        xaxis2_title='Date',
        yaxis2_title='MACD',
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_stochastic(df, ticker):
    """Plot Stochastic Oscillator"""
    k, d = calculate_stochastic(df['High'], df['Low'], df['Close'])
    
    fig = go.Figure()
    
    # Add K and D lines
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=k,
            mode='lines',
            name='%K',
            line=dict(color='blue', width=1.5)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=d,
            mode='lines',
            name='%D',
            line=dict(color='red', width=1.5)
        )
    )
    
    # Add overbought and oversold lines
    fig.add_shape(
        type="line",
        x0=df['Date'].iloc[0],
        y0=80,
        x1=df['Date'].iloc[-1],
        y1=80,
        line=dict(color="red", width=1, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=df['Date'].iloc[0],
        y0=20,
        x1=df['Date'].iloc[-1],
        y1=20,
        line=dict(color="green", width=1, dash="dash"),
    )
    
    # Add labels for the lines
    fig.add_annotation(
        x=df['Date'].iloc[-1],
        y=80,
        text="Overbought (80)",
        showarrow=False,
        yshift=10,
        font=dict(size=10, color="red")
    )
    
    fig.add_annotation(
        x=df['Date'].iloc[-1],
        y=20,
        text="Oversold (20)",
        showarrow=False,
        yshift=-10,
        font=dict(size=10, color="green")
    )
    
    fig.update_layout(
        title=f'{ticker} Stochastic Oscillator (14, 3)',
        xaxis_title='Date',
        yaxis_title='Stochastic',
        yaxis=dict(range=[0, 100]),
        height=300,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def run():
    """
    Main function to run the Technical Indicators page
    """
    st.set_page_config(
        page_title="Technical Indicators | Trading Bot",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Technical Indicators Dashboard")
    
    st.markdown("""
    This dashboard displays various technical indicators for any selected stock or trading instrument.
    Technical indicators are mathematical calculations based on price, volume, or open interest used to forecast future price movements.
    """)
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    
    # API Credentials section
    with st.sidebar.expander("Alpaca API Credentials", expanded=False):
        # Check if credentials are already in session state
        if 'alpaca_api_key' in st.session_state and 'alpaca_api_secret' in st.session_state:
            api_key = st.session_state['alpaca_api_key']
            api_secret = st.session_state['alpaca_api_secret']
            st.success("API credentials are set.")
            
            # Add option to update credentials
            update_credentials = st.checkbox("Update credentials")
            if update_credentials:
                new_api_key = st.text_input("Alpaca API Key", type="password")
                new_api_secret = st.text_input("Alpaca API Secret", type="password")
                if st.button("Save New Credentials"):
                    if new_api_key and new_api_secret:
                        st.session_state['alpaca_api_key'] = new_api_key
                        st.session_state['alpaca_api_secret'] = new_api_secret
                        st.success("API credentials updated.")
                    else:
                        st.error("Both API key and secret are required.")
        else:
            # Try to get from environment variables or secrets
            api_key = os.environ.get('ALPACA_API_KEY')
            api_secret = os.environ.get('ALPACA_API_SECRET')
            
            if api_key and api_secret:
                st.success("API credentials loaded from environment.")
                if st.button("Use these credentials"):
                    st.session_state['alpaca_api_key'] = api_key
                    st.session_state['alpaca_api_secret'] = api_secret
            else:
                # No credentials found, ask user to input
                st.warning("No API credentials found. Please enter your Alpaca API credentials.")
                input_api_key = st.text_input("Alpaca API Key", type="password")
                input_api_secret = st.text_input("Alpaca API Secret", type="password")
                if st.button("Save Credentials"):
                    if input_api_key and input_api_secret:
                        st.session_state['alpaca_api_key'] = input_api_key
                        st.session_state['alpaca_api_secret'] = input_api_secret
                        st.success("API credentials saved.")
                    else:
                        st.error("Both API key and secret are required.")
    
    # Symbol input
    ticker = st.sidebar.text_input("Enter Symbol", "AAPL").upper()
    
    # Display stock information
    if ticker:
        stock_info = get_stock_info(ticker)
        if stock_info:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                ### {stock_info['name']} ({ticker})
                **Exchange:** {stock_info['exchange']} | **Sector:** {stock_info.get('sector', 'N/A')}
                """)
            with col2:
                price = stock_info['current_price']
                change = stock_info['day_change']
                color = "green" if change >= 0 else "red"
                st.markdown(f"""
                ### ${price:.2f} <span style="color:{color};">({change:+.2f}%)</span>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"Could not find information for symbol: {ticker}")
    
    # Time period selection
    period = st.sidebar.slider("Days of Data", min_value=30, max_value=365, value=120, step=30)
    
    # Fetch real market data for this symbol
    df = get_market_data_for_indicators(ticker, period)
    
    # Indicator type selection
    indicator_categories = [
        "Moving Averages", 
        "Oscillators", 
        "Volatility", 
        "Volume", 
        "Trend", 
        "Support/Resistance",
        "Custom"
    ]
    
    category = st.sidebar.radio("Indicator Category", indicator_categories)
    
    # Available indicators based on category
    if category == "Moving Averages":
        indicators = [
            "All Moving Averages",
            "Simple Moving Average (SMA)",
            "Exponential Moving Average (EMA)",
            "Weighted Moving Average (WMA)",
            "Hull Moving Average (HMA)",
            "Volume Weighted Average Price (VWAP)",
            "Moving Average Convergence Divergence (MACD)"
        ]
    elif category == "Oscillators":
        indicators = [
            "All Oscillators",
            "Relative Strength Index (RSI)",
            "Stochastic Oscillator",
            "Williams %R",
            "Commodity Channel Index (CCI)",
            "Money Flow Index (MFI)",
            "Rate of Change (ROC)"
        ]
    elif category == "Volatility":
        indicators = [
            "All Volatility Indicators",
            "Bollinger Bands",
            "Average True Range (ATR)",
            "Keltner Channel",
            "Standard Deviation"
        ]
    elif category == "Volume":
        indicators = [
            "All Volume Indicators",
            "On-Balance Volume (OBV)",
            "Volume Profile",
            "Accumulation/Distribution Line",
            "Chaikin Money Flow"
        ]
    elif category == "Trend":
        indicators = [
            "All Trend Indicators",
            "Average Directional Index (ADX)",
            "Parabolic SAR",
            "Ichimoku Cloud",
            "Supertrend"
        ]
    elif category == "Support/Resistance":
        indicators = [
            "All Support/Resistance",
            "Pivot Points",
            "Fibonacci Retracement",
            "Support & Resistance Levels"
        ]
    else:  # Custom
        indicators = [
            "Custom Combination",
            "Triple Indicator View",
            "Multi-Timeframe Analysis"
        ]
    
    selected_indicator = st.sidebar.selectbox("Select Indicator", indicators)
    
    # Advanced Parameters (optional)
    with st.sidebar.expander("Advanced Parameters"):
        if "Moving Average" in selected_indicator or "MACD" in selected_indicator:
            sma_period = st.slider("SMA Period", min_value=5, max_value=200, value=20)
            ema_period = st.slider("EMA Period", min_value=5, max_value=200, value=20)
            if "MACD" in selected_indicator:
                macd_fast = st.slider("MACD Fast Period", min_value=5, max_value=30, value=12)
                macd_slow = st.slider("MACD Slow Period", min_value=10, max_value=50, value=26)
                macd_signal = st.slider("MACD Signal Period", min_value=3, max_value=20, value=9)
        
        elif "RSI" in selected_indicator:
            rsi_period = st.slider("RSI Period", min_value=5, max_value=30, value=14)
            rsi_overbought = st.slider("RSI Overbought Level", min_value=50, max_value=90, value=70)
            rsi_oversold = st.slider("RSI Oversold Level", min_value=10, max_value=50, value=30)
        
        elif "Stochastic" in selected_indicator:
            stoch_k = st.slider("Stochastic %K Period", min_value=5, max_value=30, value=14)
            stoch_d = st.slider("Stochastic %D Period", min_value=1, max_value=20, value=3)
            stoch_overbought = st.slider("Stochastic Overbought", min_value=50, max_value=90, value=80)
            stoch_oversold = st.slider("Stochastic Oversold", min_value=10, max_value=50, value=20)
        
        elif "Bollinger" in selected_indicator:
            bb_period = st.slider("Bollinger Band Period", min_value=5, max_value=50, value=20)
            bb_std = st.slider("Bollinger Band Standard Deviation", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    
    # Display main price chart first
    st.subheader("Price Chart")
    price_chart = plot_price_chart(df, ticker)
    st.plotly_chart(price_chart, use_container_width=True)
    
    # Display selected indicators
    if "All Moving Averages" in selected_indicator or "Moving Average" in selected_indicator:
        st.subheader("Moving Averages")
        ma_chart = plot_moving_averages(df, ticker)
        st.plotly_chart(ma_chart, use_container_width=True)
        
        # Explanation
        with st.expander("Moving Averages Explanation"):
            st.markdown("""
            ### Moving Averages
            
            Moving averages smooth out price data to create a single flowing line, making it easier to identify the direction of the trend. They are trend-following indicators and lag behind price movements.
            
            - **Simple Moving Average (SMA)**: Calculates the average price over a specific number of periods, giving equal weight to each price point.
            - **Exponential Moving Average (EMA)**: Places more weight on recent prices, making it more responsive to new information.
            
            ### Common Uses:
            1. **Trend Identification**: When price stays above a moving average, it generally indicates an uptrend. When price remains below a moving average, it suggests a downtrend.
            2. **Support and Resistance**: Moving averages can act as dynamic support in uptrends and resistance in downtrends.
            3. **Crossovers**: When a shorter-term MA crosses above a longer-term MA, it generates a bullish signal (Golden Cross), and when it crosses below, it creates a bearish signal (Death Cross).
            
            ### Common Moving Average Periods:
            - **Short-term**: 10, 20, 50 days
            - **Long-term**: 100, 200 days
            
            ### Golden Cross and Death Cross
            - **Golden Cross**: A bullish signal when a short-term MA crosses above a long-term MA (often 50-day crossing above 200-day)
            - **Death Cross**: A bearish signal when a short-term MA crosses below a long-term MA
            """)
    
    if "Bollinger" in selected_indicator or "All Volatility" in selected_indicator:
        st.subheader("Bollinger Bands")
        bb_chart = plot_bollinger_bands(df, ticker)
        st.plotly_chart(bb_chart, use_container_width=True)
        
        # Explanation
        with st.expander("Bollinger Bands Explanation"):
            st.markdown("""
            ### Bollinger Bands
            
            Bollinger Bands consist of a middle band (usually a 20-period Simple Moving Average) and an upper and lower band. The upper and lower bands are typically set 2 standard deviations away from the middle band.
            
            ### Components:
            - **Middle Band**: 20-period SMA
            - **Upper Band**: Middle Band + (2 × 20-period Standard Deviation)
            - **Lower Band**: Middle Band - (2 × 20-period Standard Deviation)
            
            ### Common Uses:
            1. **Volatility Measure**: The width of the bands increases with higher volatility and decreases with lower volatility.
            2. **Overbought/Oversold Indicator**: Price touching or exceeding the upper band may suggest an overbought condition, while touching or falling below the lower band might indicate an oversold condition.
            3. **Trend Continuation**: During strong trends, prices may "walk the band," moving along either the upper or lower band.
            4. **Bollinger Band Squeeze**: When the bands contract significantly, it often precedes a period of high volatility and potential breakout.
            
            ### Trading Signals:
            - **Bollinger Bounce**: In ranging markets, prices tend to bounce between the bands.
            - **Bollinger Squeeze**: When bands narrow significantly, prepare for a potential breakout.
            - **W-Bottoms and M-Tops**: Double bottom or top patterns that form near the lower or upper band.
            """)
    
    if "RSI" in selected_indicator or "All Oscillators" in selected_indicator:
        st.subheader("Relative Strength Index (RSI)")
        rsi_chart = plot_rsi(df, ticker)
        st.plotly_chart(rsi_chart, use_container_width=True)
        
        # Explanation
        with st.expander("RSI Explanation"):
            st.markdown("""
            ### Relative Strength Index (RSI)
            
            The RSI is a momentum oscillator that measures the speed and change of price movements. It oscillates between 0 and 100 and is typically used to identify overbought or oversold conditions.
            
            ### Calculation:
            RSI = 100 - (100 / (1 + RS))
            where RS = Average Gain / Average Loss over a specified period (typically 14 days)
            
            ### Interpretation:
            - **Overbought**: RSI > 70 suggests the asset may be overbought and due for a pullback
            - **Oversold**: RSI < 30 suggests the asset may be oversold and due for a bounce
            - **Centerline (50)**: Provides information about the underlying trend. RSI above 50 suggests bullish momentum, while RSI below 50 indicates bearish momentum.
            
            ### Trading Signals:
            1. **Divergence**: When price makes a new high/low but RSI fails to confirm with its own new high/low, it suggests potential trend reversal.
            2. **Failure Swings**: Tops and bottoms that occur without crossing overbought or oversold levels.
            3. **Support/Resistance**: RSI can show support and resistance levels that aren't visible on the price chart.
            
            ### Special Conditions:
            - During strong uptrends, RSI can remain in overbought territory for extended periods
            - During strong downtrends, RSI can remain in oversold territory for extended periods
            """)
    
    if "MACD" in selected_indicator or "All Moving Averages" in selected_indicator:
        st.subheader("Moving Average Convergence Divergence (MACD)")
        macd_chart = plot_macd(df, ticker)
        st.plotly_chart(macd_chart, use_container_width=True)
        
        # Explanation
        with st.expander("MACD Explanation"):
            st.markdown("""
            ### Moving Average Convergence Divergence (MACD)
            
            MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price.
            
            ### Components:
            - **MACD Line**: Difference between 12-period and 26-period EMA
            - **Signal Line**: 9-period EMA of the MACD Line
            - **Histogram**: Difference between MACD Line and Signal Line
            
            ### Trading Signals:
            1. **Crossovers**: When the MACD Line crosses above the Signal Line, it's a bullish signal. When it crosses below, it's a bearish signal.
            2. **Divergence**: When the price diverges from the MACD, it can signal a potential reversal.
            3. **Dramatic Rise/Fall**: When the MACD rises or falls dramatically, it indicates overbought or oversold conditions.
            4. **Zero Line Crossover**: The MACD crossing above zero is bullish; crossing below zero is bearish.
            
            ### Histogram Analysis:
            - **Increasing Histogram**: Shows increasing bullish momentum when above zero, or increasing bearish momentum when below zero.
            - **Decreasing Histogram**: Shows decreasing momentum and potential trend weakening.
            
            ### Popular Variations:
            - **MACD Histogram with Double Zero Line**
            - **MACD Histogram with Moving Average**
            - **Percentage Price Oscillator (PPO)**: Similar to MACD but uses percentages
            """)
    
    if "Stochastic" in selected_indicator or "All Oscillators" in selected_indicator:
        st.subheader("Stochastic Oscillator")
        stoch_chart = plot_stochastic(df, ticker)
        st.plotly_chart(stoch_chart, use_container_width=True)
        
        # Explanation
        with st.expander("Stochastic Oscillator Explanation"):
            st.markdown("""
            ### Stochastic Oscillator
            
            The Stochastic Oscillator is a momentum indicator that compares a particular closing price of a security to a range of its prices over a certain period of time. It follows the speed or momentum of price and oscillates between 0 and 100.
            
            ### Components:
            - **%K Line**: The main line showing the current price relative to the high-low range over a set number of periods.
            - **%D Line**: A moving average of %K, which acts as a signal line.
            
            ### Calculation:
            - **%K = (Current Close - Lowest Low) / (Highest High - Lowest Low) × 100**
            - **%D = 3-day SMA of %K**
            
            ### Interpretation:
            - **Overbought**: Readings above 80 suggest overbought conditions
            - **Oversold**: Readings below 20 suggest oversold conditions
            - **Crossovers**: When %K crosses above %D, it's a bullish signal. When it crosses below, it's a bearish signal.
            
            ### Trading Signals:
            1. **Overbought/Oversold Reversals**: Look for the Stochastic to reverse from overbought/oversold territory
            2. **Divergence**: When price makes a new high/low but Stochastic doesn't confirm
            3. **Bull/Bear Setup**: When both %K and %D are below 20 and then cross above 20 (bullish setup), or when both are above 80 and then cross below 80 (bearish setup)
            
            ### Variations:
            - **Fast Stochastic**: Uses raw %K and 3-period SMA for %D
            - **Slow Stochastic**: Uses 3-period SMA of %K as %K, and 3-period SMA of that as %D (reduces sensitivity)
            - **Full Stochastic**: Allows customization of all parameters
            """)
    
    # Numerical indicators section
    st.subheader("Current Technical Indicator Values")
    
    # Calculate current indicator values
    current_data = {
        "Symbol": ticker,
        "Last Close Price": f"${df['Close'].iloc[-1]:.2f}",
        "SMA (20)": f"${calculate_sma(df['Close'], 20).iloc[-1]:.2f}",
        "SMA (50)": f"${calculate_sma(df['Close'], 50).iloc[-1]:.2f}",
        "SMA (200)": f"${calculate_sma(df['Close'], 200).iloc[-1]:.2f}",
        "EMA (12)": f"${calculate_ema(df['Close'], 12).iloc[-1]:.2f}",
        "EMA (26)": f"${calculate_ema(df['Close'], 26).iloc[-1]:.2f}",
        "RSI (14)": f"{calculate_rsi(df['Close'], 14).iloc[-1]:.2f}",
        "MACD Line": f"{calculate_macd(df['Close'])[0].iloc[-1]:.4f}",
        "MACD Signal": f"{calculate_macd(df['Close'])[1].iloc[-1]:.4f}",
        "MACD Histogram": f"{calculate_macd(df['Close'])[2].iloc[-1]:.4f}",
        "Bollinger Upper": f"${calculate_bollinger_bands(df['Close'])[0].iloc[-1]:.2f}",
        "Bollinger Middle": f"${calculate_bollinger_bands(df['Close'])[1].iloc[-1]:.2f}",
        "Bollinger Lower": f"${calculate_bollinger_bands(df['Close'])[2].iloc[-1]:.2f}",
        "Stochastic %K": f"{calculate_stochastic(df['High'], df['Low'], df['Close'])[0].iloc[-1]:.2f}",
        "Stochastic %D": f"{calculate_stochastic(df['High'], df['Low'], df['Close'])[1].iloc[-1]:.2f}"
    }
    
    # Create columns for indicators display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Price & Moving Averages")
        st.info(f"Last Close: {current_data['Last Close Price']}")
        st.markdown(f"SMA (20): {current_data['SMA (20)']}")
        st.markdown(f"SMA (50): {current_data['SMA (50)']}")
        st.markdown(f"SMA (200): {current_data['SMA (200)']}")
        st.markdown(f"EMA (12): {current_data['EMA (12)']}")
        st.markdown(f"EMA (26): {current_data['EMA (26)']}")
    
    with col2:
        st.markdown("### Oscillators")
        
        # RSI with color
        rsi_value = float(current_data['RSI (14)'])
        rsi_color = "red" if rsi_value > 70 else "green" if rsi_value < 30 else "black"
        rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
        st.markdown(f"RSI (14): <span style='color:{rsi_color}'>{current_data['RSI (14)']} ({rsi_status})</span>", unsafe_allow_html=True)
        
        # Stochastic with color
        stoch_k = float(current_data['Stochastic %K'])
        stoch_color = "red" if stoch_k > 80 else "green" if stoch_k < 20 else "black"
        stoch_status = "Overbought" if stoch_k > 80 else "Oversold" if stoch_k < 20 else "Neutral"
        st.markdown(f"Stochastic %K: <span style='color:{stoch_color}'>{current_data['Stochastic %K']} ({stoch_status})</span>", unsafe_allow_html=True)
        st.markdown(f"Stochastic %D: {current_data['Stochastic %D']}")
        
        # MACD with color
        macd_hist = float(current_data['MACD Histogram'])
        macd_color = "green" if macd_hist > 0 else "red"
        macd_signal = "Bullish" if macd_hist > 0 else "Bearish"
        st.markdown(f"MACD Histogram: <span style='color:{macd_color}'>{current_data['MACD Histogram']} ({macd_signal})</span>", unsafe_allow_html=True)
        st.markdown(f"MACD Line: {current_data['MACD Line']}")
        st.markdown(f"MACD Signal: {current_data['MACD Signal']}")
    
    with col3:
        st.markdown("### Volatility & Bands")
        
        # Get current price
        current_price = df['Close'].iloc[-1]
        
        # Bollinger Bands position
        bb_upper = float(current_data['Bollinger Upper'].replace('$', ''))
        bb_lower = float(current_data['Bollinger Lower'].replace('$', ''))
        bb_middle = float(current_data['Bollinger Middle'].replace('$', ''))
        
        bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower)) * 100
        bb_status = "Upper Band" if bb_position > 80 else "Lower Band" if bb_position < 20 else "Middle Band"
        bb_color = "red" if bb_position > 80 else "green" if bb_position < 20 else "blue"
        
        st.markdown(f"BB Position: <span style='color:{bb_color}'>{bb_position:.1f}% ({bb_status})</span>", unsafe_allow_html=True)
        st.markdown(f"BB Upper: {current_data['Bollinger Upper']}")
        st.markdown(f"BB Middle: {current_data['Bollinger Middle']}")
        st.markdown(f"BB Lower: {current_data['Bollinger Lower']}")
        
        # Calculate BB Width
        bb_width = ((bb_upper - bb_lower) / bb_middle) * 100
        st.markdown(f"BB Width: {bb_width:.2f}%")
    
    # Signal Summary based on indicators
    st.subheader("Technical Signal Summary")
    
    # Calculate signals
    ma_signal = "Bullish" if current_price > float(current_data['SMA (50)'].replace('$', '')) and float(current_data['SMA (50)'].replace('$', '')) > float(current_data['SMA (200)'].replace('$', '')) else "Bearish" if current_price < float(current_data['SMA (50)'].replace('$', '')) and float(current_data['SMA (50)'].replace('$', '')) < float(current_data['SMA (200)'].replace('$', '')) else "Neutral"
    
    rsi_signal = "Bearish (Overbought)" if rsi_value > 70 else "Bullish (Oversold)" if rsi_value < 30 else "Neutral"
    
    macd_signal = "Bullish" if macd_hist > 0 else "Bearish"
    
    stoch_signal = "Bearish (Overbought)" if stoch_k > 80 else "Bullish (Oversold)" if stoch_k < 20 else "Neutral"
    
    bb_signal = "Bearish (Upper Band)" if bb_position > 80 else "Bullish (Lower Band)" if bb_position < 20 else "Neutral"
    
    # Count bullish and bearish signals
    signals = [ma_signal, rsi_signal, macd_signal, stoch_signal, bb_signal]
    bullish_count = sum(1 for s in signals if "Bullish" in s)
    bearish_count = sum(1 for s in signals if "Bearish" in s)
    neutral_count = sum(1 for s in signals if "Neutral" in s)
    
    # Determine overall signal
    if bullish_count > bearish_count and bullish_count > neutral_count:
        overall_signal = "Bullish"
        signal_color = "green"
    elif bearish_count > bullish_count and bearish_count > neutral_count:
        overall_signal = "Bearish"
        signal_color = "red"
    else:
        overall_signal = "Neutral"
        signal_color = "gray"
    
    # Display signals in a visual way
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; 
                    background-color: {'rgba(0,128,0,0.2)' if overall_signal == 'Bullish' else 
                                        'rgba(255,0,0,0.2)' if overall_signal == 'Bearish' else 
                                        'rgba(200,200,200,0.2)'};
                    border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin: 0; color: {signal_color};">
                {overall_signal} Signal
            </h2>
            <p style="margin: 5px 0 0 0;">
                Based on {bullish_count} bullish, {bearish_count} bearish, and {neutral_count} neutral indicators
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display individual signals
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        ma_color = "green" if "Bullish" in ma_signal else "red" if "Bearish" in ma_signal else "gray"
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: rgba({','.join(['0', '128', '0'] if "Bullish" in ma_signal else ['255', '0', '0'] if "Bearish" in ma_signal else ['200', '200', '200'])},0.1); border-radius: 5px;">
            <h4 style="margin: 0; color: {ma_color};">Moving Averages</h4>
            <p style="margin: 5px 0 0 0; color: {ma_color};">{ma_signal}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rsi_color = "green" if "Bullish" in rsi_signal else "red" if "Bearish" in rsi_signal else "gray"
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: rgba({','.join(['0', '128', '0'] if "Bullish" in rsi_signal else ['255', '0', '0'] if "Bearish" in rsi_signal else ['200', '200', '200'])},0.1); border-radius: 5px;">
            <h4 style="margin: 0; color: {rsi_color};">RSI</h4>
            <p style="margin: 5px 0 0 0; color: {rsi_color};">{rsi_signal}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        macd_color = "green" if "Bullish" in macd_signal else "red" if "Bearish" in macd_signal else "gray"
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: rgba({','.join(['0', '128', '0'] if "Bullish" in macd_signal else ['255', '0', '0'] if "Bearish" in macd_signal else ['200', '200', '200'])},0.1); border-radius: 5px;">
            <h4 style="margin: 0; color: {macd_color};">MACD</h4>
            <p style="margin: 5px 0 0 0; color: {macd_color};">{macd_signal}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        stoch_color = "green" if "Bullish" in stoch_signal else "red" if "Bearish" in stoch_signal else "gray"
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: rgba({','.join(['0', '128', '0'] if "Bullish" in stoch_signal else ['255', '0', '0'] if "Bearish" in stoch_signal else ['200', '200', '200'])},0.1); border-radius: 5px;">
            <h4 style="margin: 0; color: {stoch_color};">Stochastic</h4>
            <p style="margin: 5px 0 0 0; color: {stoch_color};">{stoch_signal}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        bb_color = "green" if "Bullish" in bb_signal else "red" if "Bearish" in bb_signal else "gray"
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: rgba({','.join(['0', '128', '0'] if "Bullish" in bb_signal else ['255', '0', '0'] if "Bearish" in bb_signal else ['200', '200', '200'])},0.1); border-radius: 5px;">
            <h4 style="margin: 0; color: {bb_color};">Bollinger Bands</h4>
            <p style="margin: 5px 0 0 0; color: {bb_color};">{bb_signal}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional Resources
    with st.expander("Additional Resources & Tutorials"):
        st.markdown("""
        ### Learning Technical Analysis
        
        #### Books
        - "Technical Analysis of the Financial Markets" by John J. Murphy
        - "Getting Started in Technical Analysis" by Jack D. Schwager
        - "Encyclopedia of Chart Patterns" by Thomas Bulkowski
        
        #### Websites
        - [Investopedia Technical Analysis](https://www.investopedia.com/technical-analysis-4689657)
        - [StockCharts.com ChartSchool](https://school.stockcharts.com/)
        - [BabyPips.com (for Forex)](https://www.babypips.com/learn/forex/technical-analysis)
        
        #### Important Concepts
        1. **Trend Analysis**: Identifying the direction of the market's movement
        2. **Support and Resistance**: Price levels where a stock historically has difficulty falling below (support) or rising above (resistance)
        3. **Chart Patterns**: Formations on charts that suggest potential continuations or reversals
        4. **Indicators and Oscillators**: Mathematical calculations that help determine overbought/oversold conditions and trend strength
        5. **Volume Analysis**: Study of trading volume to confirm price movements
        
        #### Common Mistakes to Avoid
        - Relying on a single indicator
        - Ignoring the broader market trends
        - Over-analyzing (paralysis by analysis)
        - Failing to use stop-loss orders
        - Ignoring fundamental factors
        """)

# Execute the main function
if __name__ == "__main__":
    run()