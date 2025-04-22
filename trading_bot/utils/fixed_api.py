"""
Fixed API module to directly connect with Alpaca
"""
import streamlit as st
from .alpaca_api import (
    test_alpaca_connection as direct_test_connection,
    get_account_info as direct_get_account_info,
    get_portfolio as direct_get_portfolio,
    get_market_data as direct_get_market_data
)

# Keep original import structure to maintain compatibility with legacy code
from .api import (
    get_fingpt_news, get_fingpt_signals, get_asset_price,
    get_historical_data, place_order
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

def get_orders():
    """Get orders (placeholder until direct implementation)"""
    # Currently falls back to the api.py implementation 
    # This is a stub for future direct implementation
    from .api import get_orders as original_get_orders
    return original_get_orders()