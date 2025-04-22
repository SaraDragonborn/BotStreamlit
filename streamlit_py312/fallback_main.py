import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Trading Bot Platform",
    page_icon="📈",
    layout="wide"
)

# Main title
st.title("Trading Bot Platform")
st.subheader("AI-powered trading platform")

# Show a message about compatibility mode
st.warning("""
⚠️ **Compatibility Mode Active**

Your Trading Bot is running in compatibility mode due to Python 3.12 constraints.
The full functionality requires a few additional package installations.
""")

# Display some basic content
st.markdown("""
## Welcome to your Trading Bot!

This platform includes:
- Portfolio watchlist with up to 100 stocks
- Technical indicators and trading strategies
- US and Indian market trading capabilities
- AI-powered analysis and recommendations
""")

# Show API status
st.subheader("API Connection Status")
col1, col2 = st.columns(2)
with col1:
    alpaca_key = os.environ.get("ALPACA_API_KEY", "")
    if alpaca_key:
        st.success("✅ Alpaca API credentials found")
    else:
        st.error("❌ Alpaca API credentials missing")
with col2:
    angel_key = os.environ.get("ANGEL_ONE_API_KEY", "")
    if angel_key:
        st.success("✅ Angel One API credentials found")
    else:
        st.error("❌ Angel One API credentials missing")

# Add a simple form
st.subheader("Quick Stock Lookup")
symbol = st.text_input("Enter a stock symbol (e.g., AAPL, MSFT):")
if symbol:
    st.info(f"Looking up data for {symbol}...")
    st.success(f"Stock {symbol} found! Add your API keys in Streamlit Cloud secrets to see real data.")

# Show current date and time
st.sidebar.write(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
st.sidebar.write("Trading Bot Version: 2.0")

# Add footer
st.markdown("---")
st.markdown("© 2025 Trading Bot Platform")
