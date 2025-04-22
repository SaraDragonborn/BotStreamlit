"""
Fixed API module to directly connect with Alpaca
"""
import streamlit as st
from .alpaca_api import (
    test_alpaca_connection as direct_test_connection,
    get_account_info as direct_get_account_info,
    get_portfolio as direct_get_portfolio,
    get_orders as direct_get_orders,
    place_order as direct_place_order,
    get_market_data as direct_get_market_data
)

# Keep original import structure to maintain compatibility with legacy code
from .api import (
    get_fingpt_news, get_fingpt_signals, get_asset_price,
    get_historical_data
)

def test_alpaca_connection(api_key=None, api_secret=None):
    """Test connection to Alpaca API"""
    # Use keys from parameters or session state
    if not api_key and 'alpaca_api_key' in st.session_state:
        api_key = st.session_state.alpaca_api_key
    
    if not api_secret and 'alpaca_api_secret' in st.session_state:
        api_secret = st.session_state.alpaca_api_secret
    
    mode = st.session_state.get('trading_mode', 'paper')
    
    # Call direct connection function
    return direct_test_connection(api_key, api_secret, mode)

def get_account_info():
    """Get account info using direct Alpaca connection"""
    if 'alpaca_api_key' not in st.session_state or not st.session_state.alpaca_api_key:
        return None
    
    api_key = st.session_state.alpaca_api_key
    api_secret = st.session_state.alpaca_api_secret
    paper = st.session_state.get('trading_mode', 'paper') == 'paper'
    
    return direct_get_account_info(api_key, api_secret, paper)

def get_positions():
    """Get positions using direct Alpaca connection"""
    if 'alpaca_api_key' not in st.session_state or not st.session_state.alpaca_api_key:
        return []
    
    api_key = st.session_state.alpaca_api_key
    api_secret = st.session_state.alpaca_api_secret
    paper = st.session_state.get('trading_mode', 'paper') == 'paper'
    
    return direct_get_portfolio(api_key, api_secret, paper)

def get_orders(status='all'):
    """Get orders using direct Alpaca connection"""
    if 'alpaca_api_key' not in st.session_state or not st.session_state.alpaca_api_key:
        return []
    
    api_key = st.session_state.alpaca_api_key
    api_secret = st.session_state.alpaca_api_secret
    paper = st.session_state.get('trading_mode', 'paper') == 'paper'
    
    return direct_get_orders(api_key, api_secret, paper, status)

def place_order(symbol, qty, side, type="market", time_in_force="day", limit_price=None, stop_price=None):
    """Place an order using direct Alpaca connection"""
    if 'alpaca_api_key' not in st.session_state or not st.session_state.alpaca_api_key:
        return {"error": "API credentials not found"}
    
    api_key = st.session_state.alpaca_api_key
    api_secret = st.session_state.alpaca_api_secret
    paper = st.session_state.get('trading_mode', 'paper') == 'paper'
    
    return direct_place_order(
        api_key, api_secret, symbol, qty, side, 
        type, time_in_force, limit_price, stop_price, paper
    )