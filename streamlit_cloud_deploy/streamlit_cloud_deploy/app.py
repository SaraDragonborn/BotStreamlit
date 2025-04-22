import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add utils directory to path if not already added
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "utils"))
if utils_path not in sys.path:
    sys.path.append(utils_path)

# Import utilities
try:
    from utils.credentials import get_alpaca_credentials, get_angel_one_credentials
except ImportError:
    # Fallback if imports fail
    def get_alpaca_credentials():
        return None
    def get_angel_one_credentials():
        return None

# Configure the page
st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1E88E5;
    margin-bottom: 1rem;
}
.card {
    background-color: #f5f7f9;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Main title
st.title("AI Trading Bot")
st.subheader("Advanced Trading Platform with Artificial Intelligence")

# Define a function to fetch stock data
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_stock_data(symbol, period="1y"):
    """Fetch stock data using yfinance"""
    try:
        data = yf.download(symbol, period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Initialize session state for watchlist if it doesn't exist
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Initialize API key session state
if 'alpaca_api_key' not in st.session_state:
    st.session_state.alpaca_api_key = ""
if 'alpaca_api_secret' not in st.session_state:
    st.session_state.alpaca_api_secret = ""
if 'angel_api_key' not in st.session_state:
    st.session_state.angel_api_key = ""
if 'angel_client_id' not in st.session_state:
    st.session_state.angel_client_id = ""
if 'angel_password' not in st.session_state:
    st.session_state.angel_password = ""

# Check API credentials
alpaca_credentials = get_alpaca_credentials()
angel_one_credentials = get_angel_one_credentials()

# Main dashboard
st.markdown('<p class="main-header">Market Dashboard</p>', unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Market Overview", "Watchlist", "Quick Trade"])

with tab1:
    st.header("Market Overview")
    
    # Create columns for market indices
    col1, col2, col3, col4 = st.columns(4)
    
    # S&P 500
    with col1:
        with st.container():
            sp500 = get_stock_data("^GSPC", "1mo")
            if sp500 is not None and not sp500.empty:
                latest_close = sp500['Close'].iloc[-1]
                prev_close = sp500['Close'].iloc[-2]
                change = latest_close - prev_close
                change_pct = (change / prev_close) * 100
                st.metric("S&P 500", f"${latest_close:.2f}", f"{change_pct:.2f}%")
            else:
                st.metric("S&P 500", "Loading...", "")
    
    # Dow Jones
    with col2:
        with st.container():
            dow = get_stock_data("^DJI", "1mo")
            if dow is not None and not dow.empty:
                latest_close = dow['Close'].iloc[-1]
                prev_close = dow['Close'].iloc[-2]
                change = latest_close - prev_close
                change_pct = (change / prev_close) * 100
                st.metric("Dow Jones", f"${latest_close:.2f}", f"{change_pct:.2f}%")
            else:
                st.metric("Dow Jones", "Loading...", "")
    
    # NASDAQ
    with col3:
        with st.container():
            nasdaq = get_stock_data("^IXIC", "1mo")
            if nasdaq is not None and not nasdaq.empty:
                latest_close = nasdaq['Close'].iloc[-1]
                prev_close = nasdaq['Close'].iloc[-2]
                change = latest_close - prev_close
                change_pct = (change / prev_close) * 100
                st.metric("NASDAQ", f"${latest_close:.2f}", f"{change_pct:.2f}%")
            else:
                st.metric("NASDAQ", "Loading...", "")
    
    # VIX
    with col4:
        with st.container():
            vix = get_stock_data("^VIX", "1mo")
            if vix is not None and not vix.empty:
                latest_close = vix['Close'].iloc[-1]
                prev_close = vix['Close'].iloc[-2]
                change = latest_close - prev_close
                change_pct = (change / prev_close) * 100
                st.metric("VIX", f"{latest_close:.2f}", f"{change_pct:.2f}%")
            else:
                st.metric("VIX", "Loading...", "")
    
    # Display S&P 500 chart
    st.subheader("S&P 500 Performance")
    sp500 = get_stock_data("^GSPC", "6mo")
    
    if sp500 is not None and not sp500.empty:
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=sp500.index,
            open=sp500['Open'],
            high=sp500['High'],
            low=sp500['Low'],
            close=sp500['Close'],
            name="S&P 500"
        ))
        
        # Add moving averages
        ma50 = sp500['Close'].rolling(window=50).mean()
        ma200 = sp500['Close'].rolling(window=200).mean()
        
        fig.add_trace(go.Scatter(
            x=sp500.index,
            y=ma50,
            line=dict(color='blue', width=1),
            name="50-day MA"
        ))
        
        fig.add_trace(go.Scatter(
            x=sp500.index,
            y=ma200,
            line=dict(color='red', width=1),
            name="200-day MA"
        ))
        
        # Update layout
        fig.update_layout(
            title="S&P 500 with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            xaxis_rangeslider_visible=False,
            template="plotly_white"
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Loading market data...")

    # Market News
    st.subheader("Market News")
    
    # In a real implementation, this would fetch news from an API
    news_items = [
        {"title": "Fed Signals Potential Rate Cut", "source": "Financial Times", "date": "Today"},
        {"title": "Tech Stocks Rally on Earnings Beat", "source": "Wall Street Journal", "date": "Yesterday"},
        {"title": "Oil Prices Stabilize After Supply Concerns", "source": "Bloomberg", "date": "2 days ago"},
        {"title": "New Economic Data Shows Strong Job Growth", "source": "CNBC", "date": "3 days ago"}
    ]
    
    for news in news_items:
        st.markdown(f"**{news['title']}** - {news['source']} ({news['date']})")
    
    st.markdown("[View More News →](https://finance.yahoo.com/news/)")

with tab2:
    st.header("Watchlist")
    
    # Add stock to watchlist
    with st.form("add_to_watchlist"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            new_symbol = st.text_input("Add symbol to watchlist:")
        
        with col2:
            st.write("")
            st.write("")
            add_button = st.form_submit_button("Add")
    
    if add_button and new_symbol:
        new_symbol = new_symbol.strip().upper()
        if new_symbol not in st.session_state.watchlist:
            # Verify the symbol
            data = get_stock_data(new_symbol, "1d")
            if data is not None and not data.empty:
                st.session_state.watchlist.append(new_symbol)
                st.success(f"{new_symbol} added to watchlist")
            else:
                st.error(f"Could not verify symbol {new_symbol}")
        else:
            st.info(f"{new_symbol} is already in your watchlist")
    
    # Display watchlist
    if st.session_state.watchlist:
        # Prepare data for table
        watchlist_data = []
        
        for symbol in st.session_state.watchlist:
            data = get_stock_data(symbol, "5d")
            
            if data is not None and not data.empty:
                latest_close = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2] if len(data) > 1 else latest_close
                change = latest_close - prev_close
                change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                
                # Get additional info
                ticker = yf.Ticker(symbol)
                info = {}
                try:
                    info = ticker.info
                except:
                    pass
                
                name = info.get('shortName', symbol)
                market_cap = info.get('marketCap', 'N/A')
                if market_cap != 'N/A':
                    market_cap = f"${market_cap/1e9:.2f}B"
                
                # Add to data list
                watchlist_data.append({
                    "Symbol": symbol,
                    "Name": name,
                    "Last Price": f"${latest_close:.2f}",
                    "Change": f"${change:.2f}",
                    "Change %": f"{change_pct:.2f}%",
                    "Market Cap": market_cap,
                    "Volume": f"{data['Volume'].iloc[-1]:,.0f}",
                })
            else:
                # Add placeholder if data couldn't be loaded
                watchlist_data.append({
                    "Symbol": symbol,
                    "Name": "Loading...",
                    "Last Price": "N/A",
                    "Change": "N/A",
                    "Change %": "N/A",
                    "Market Cap": "N/A",
                    "Volume": "N/A",
                })
        
        # Create dataframe and display
        if watchlist_data:
            df = pd.DataFrame(watchlist_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Option to remove from watchlist
        symbol_to_remove = st.selectbox("Select symbol to remove", [""] + st.session_state.watchlist)
        if st.button("Remove from Watchlist") and symbol_to_remove:
            st.session_state.watchlist.remove(symbol_to_remove)
            st.experimental_rerun()
    else:
        st.info("Your watchlist is empty. Add symbols using the form above.")

with tab3:
    st.header("Quick Trade")
    
    # Symbol input
    symbol = st.text_input("Enter Symbol", "AAPL")
    
    if symbol:
        # Get stock data
        data = get_stock_data(symbol, "1mo")
        
        if data is not None and not data.empty:
            # Display stock info
            ticker = yf.Ticker(symbol)
            info = {}
            try:
                info = ticker.info
            except:
                pass
            
            name = info.get('shortName', symbol)
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            
            st.subheader(f"{name} ({symbol})")
            st.write(f"**Sector:** {sector} | **Industry:** {industry}")
            
            # Display current price and daily change
            latest_close = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2]
            change = latest_close - prev_close
            change_pct = (change / prev_close) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${latest_close:.2f}", f"{change_pct:.2f}%")
            with col2:
                st.metric("Day High", f"${data['High'].iloc[-1]:.2f}")
            with col3:
                st.metric("Day Low", f"${data['Low'].iloc[-1]:.2f}")
            
            # Display chart
            st.subheader(f"{symbol} Price Chart")
            
            # Create candlestick chart
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} - 1 Month Performance",
                xaxis_title="Date",
                yaxis_title="Price",
                height=500,
                xaxis_rangeslider_visible=False,
                template="plotly_white"
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Trading simulator
            st.subheader("Trading Simulator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                trade_type = st.radio("Trade Type", ["Buy", "Sell"])
                quantity = st.number_input("Quantity", min_value=1, value=10)
                
                price = data['Close'].iloc[-1]
                trade_value = price * quantity
                
                st.write(f"Trade Value: ${trade_value:.2f}")
                
                if st.button("Place Trade"):
                    if alpaca_credentials:
                        st.success(f"{trade_type} order for {quantity} shares of {symbol} at ${price:.2f} sent to Alpaca!")
                    else:
                        st.warning("API credentials not configured. Go to Settings to add your API keys.")
            
            with col2:
                st.subheader("Performance Metrics")
                
                # Calculate some metrics
                daily_returns = data['Close'].pct_change()
                volatility = daily_returns.std() * np.sqrt(252) * 100
                
                # Display metrics
                st.write(f"Daily Return: {daily_returns.iloc[-1]*100:.2f}%")
                st.write(f"5-Day Return: {(data['Close'].iloc[-1]/data['Close'].iloc[-5]-1)*100:.2f}%")
                st.write(f"Monthly Return: {(data['Close'].iloc[-1]/data['Close'].iloc[0]-1)*100:.2f}%")
                st.write(f"Annualized Volatility: {volatility:.2f}%")
                
                # Basic trading signals
                ma50 = data['Close'].rolling(window=20).mean().iloc[-1]
                ma200 = data['Close'].rolling(window=50).mean().iloc[-1]
                
                if ma50 > ma200:
                    st.write("Signal: 📈 Bullish (MA20 > MA50)")
                else:
                    st.write("Signal: 📉 Bearish (MA20 < MA50)")
        else:
            st.error(f"Could not load data for {symbol}")

# Sidebar
st.sidebar.title("AI Trading Bot")

# Navigation
st.sidebar.header("Navigation")
st.sidebar.markdown("""
- [Dashboard](#market-dashboard)
- [Strategies](pages/1_Strategies.py)
- [Portfolio](pages/2_Portfolio.py)
- [Settings](pages/3_Settings.py)
""")

# API Status
st.sidebar.header("API Status")
alpaca_status = "✅ Connected" if alpaca_credentials else "❌ Not Connected"
angel_status = "✅ Connected" if angel_one_credentials else "❌ Not Connected"

st.sidebar.markdown(f"**Alpaca API:** {alpaca_status}")
st.sidebar.markdown(f"**Angel One API:** {angel_status}")

# Quick API setup if credentials are missing
if not alpaca_credentials or not angel_one_credentials:
    st.sidebar.warning("API credentials not configured!")
    if st.sidebar.button("Configure APIs"):
        st.markdown(f"<a href='pages/3_Settings.py' target='_self'>Go to Settings</a>", unsafe_allow_html=True)

# About
st.sidebar.header("About")
st.sidebar.info(
    "AI Trading Bot is an advanced trading platform powered by artificial intelligence. "
    "Monitor markets, analyze stocks, and execute trades with ease."
)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"© 2025 AI Trading Bot | v1.0.0")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")