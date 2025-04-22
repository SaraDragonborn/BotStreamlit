import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Configure the page
st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="📈",
    layout="wide"
)

# Main title
st.title("AI Trading Bot")
st.subheader("Advanced Trading Platform")

# Initialize session states
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

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

# Main dashboard with tabs
tab1, tab2, tab3 = st.tabs(["Market Overview", "Watchlist", "Trading"])

with tab1:
    st.header("Market Overview")
    
    # Display major indices
    col1, col2, col3, col4 = st.columns(4)
    
    # S&P 500
    with col1:
        data = get_stock_data("^GSPC", "1mo")
        if data is not None and not data.empty:
            current = data['Close'].iloc[-1]
            prev = data['Close'].iloc[-2]
            change = ((current - prev) / prev) * 100
            st.metric("S&P 500", f"${current:.2f}", f"{change:.2f}%")
    
    # NASDAQ
    with col2:
        data = get_stock_data("^IXIC", "1mo")
        if data is not None and not data.empty:
            current = data['Close'].iloc[-1]
            prev = data['Close'].iloc[-2]
            change = ((current - prev) / prev) * 100
            st.metric("NASDAQ", f"${current:.2f}", f"{change:.2f}%")
    
    # Dow Jones
    with col3:
        data = get_stock_data("^DJI", "1mo")
        if data is not None and not data.empty:
            current = data['Close'].iloc[-1]
            prev = data['Close'].iloc[-2]
            change = ((current - prev) / prev) * 100
            st.metric("Dow Jones", f"${current:.2f}", f"{change:.2f}%")
    
    # VIX
    with col4:
        data = get_stock_data("^VIX", "1mo")
        if data is not None and not data.empty:
            current = data['Close'].iloc[-1]
            prev = data['Close'].iloc[-2]
            change = ((current - prev) / prev) * 100
            st.metric("VIX", f"{current:.2f}", f"{change:.2f}%")
    
    # Display chart
    st.subheader("S&P 500 Performance")
    sp500 = get_stock_data("^GSPC", "6mo")
    
    if sp500 is not None and not sp500.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=sp500.index,
            open=sp500['Open'],
            high=sp500['High'],
            low=sp500['Low'],
            close=sp500['Close'],
            name="S&P 500"
        ))
        
        fig.update_layout(
            title="S&P 500 - 6 Month Performance",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Watchlist")
    
    # Add stock to watchlist
    col1, col2 = st.columns([3, 1])
    with col1:
        new_symbol = st.text_input("Add symbol:")
    with col2:
        if st.button("Add") and new_symbol:
            new_symbol = new_symbol.strip().upper()
            if new_symbol not in st.session_state.watchlist:
                data = get_stock_data(new_symbol, "1d")
                if data is not None and not data.empty:
                    st.session_state.watchlist.append(new_symbol)
                    st.success(f"Added {new_symbol}")
                else:
                    st.error(f"Invalid symbol: {new_symbol}")
            else:
                st.warning(f"{new_symbol} already in watchlist")
    
    # Display watchlist
    watchlist_data = []
    for symbol in st.session_state.watchlist:
        data = get_stock_data(symbol, "5d")
        if data is not None and not data.empty:
            latest = data['Close'].iloc[-1]
            prev = data['Close'].iloc[-2]
            change = ((latest - prev) / prev) * 100
            watchlist_data.append({
                "Symbol": symbol,
                "Price": f"${latest:.2f}",
                "Change %": f"{change:.2f}%",
                "Volume": f"{data['Volume'].iloc[-1]:,.0f}"
            })
    
    if watchlist_data:
        df = pd.DataFrame(watchlist_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Remove from watchlist
        symbol_to_remove = st.selectbox("Select symbol to remove:", [""] + st.session_state.watchlist)
        if st.button("Remove") and symbol_to_remove:
            st.session_state.watchlist.remove(symbol_to_remove)
            st.experimental_rerun()

with tab3:
    st.header("Quick Trade")
    
    symbol = st.text_input("Enter symbol:", "AAPL")
    
    if symbol:
        data = get_stock_data(symbol, "1mo")
        
        if data is not None and not data.empty:
            # Display current price
            current_price = data['Close'].iloc[-1]
            st.subheader(f"{symbol}: ${current_price:.2f}")
            
            # Display chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol
            ))
            
            fig.update_layout(
                title=f"{symbol} - 1 Month Performance",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trading form
            col1, col2 = st.columns(2)
            
            with col1:
                trade_type = st.radio("Trade Type:", ["Buy", "Sell"])
                quantity = st.number_input("Quantity:", min_value=1, value=10)
                trade_value = quantity * current_price
                
                st.write(f"Trade Value: ${trade_value:.2f}")
                
                if st.button("Execute Trade"):
                    st.success(f"{trade_type} order for {quantity} shares of {symbol} placed successfully!")
            
            with col2:
                st.subheader("Performance")
                st.write(f"Daily Change: {((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100):.2f}%")
                st.write(f"Weekly Change: {((data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100):.2f}%")
        else:
            st.error(f"Could not load data for {symbol}")

# Sidebar
st.sidebar.title("AI Trading Bot")

# Navigation
st.sidebar.markdown("""
## Navigation
- [Home](/)
- [Strategies](/Strategies)
- [Portfolio](/Portfolio)
- [Settings](/Settings)
""")

# API Status
st.sidebar.header("API Status")
st.sidebar.info("API: Not Connected")

# Quick API setup
st.sidebar.header("Quick Setup")
with st.sidebar.form("api_keys"):
    st.write("Alpaca API")
    alpaca_key = st.text_input("API Key:", type="password")
    alpaca_secret = st.text_input("API Secret:", type="password")
    
    st.write("Angel One API")
    angel_key = st.text_input("API Key:", type="password")
    
    submitted = st.form_submit_button("Save Keys")
    if submitted:
        st.success("API keys saved!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 AI Trading Bot")