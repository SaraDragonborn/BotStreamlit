"""
Fixed API module to directly connect with Alpaca and Angel One
"""
import streamlit as st
from .alpaca_api import (
    test_alpaca_connection as direct_test_connection,
    get_account_info as direct_get_account_info,
    get_portfolio as direct_get_portfolio,
    get_market_data as direct_get_market_data
)
from .angel_one_api import (
    test_angel_one_connection as direct_test_angel_one_connection
)

# Import Indian market connector functions
from .indian_market_connector import (
    get_indian_market_symbols, get_indian_historical_data, run_indian_backtest,
    get_active_indian_strategies, add_indian_strategy, update_indian_strategy, 
    delete_indian_strategy, get_indian_market_data_sources
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

def test_angel_one_connection():
    """Test connection to Angel One API"""
    # Get credentials from session state
    client_id = st.session_state.get('angel_one_client_id')
    password = st.session_state.get('angel_one_password')
    api_key = st.session_state.get('angel_one_api_key')
    totp_key = st.session_state.get('angel_one_totp_key', '')
    
    # Call direct connection function
    return direct_test_angel_one_connection(client_id, password, api_key, totp_key)

# Export Indian market connector functions directly
def get_indian_symbols():
    """Get a list of available Indian market symbols"""
    return get_indian_market_symbols()

def get_indian_data(symbol, exchange="NSE", interval="ONE_DAY", days=30):
    """Get historical data for an Indian market symbol"""
    return get_indian_historical_data(symbol, exchange, interval, days)

def backtest_indian_strategy(strategy_name, symbol, exchange="NSE", interval="ONE_DAY", days=180, params=None):
    """Run backtest for an Indian market strategy"""
    return run_indian_backtest(strategy_name, symbol, exchange, interval, days, params)

def get_indian_strategies():
    """Get a list of active Indian market strategies"""
    return get_active_indian_strategies()

def add_strategy_indian(name, description, symbol, strategy_type, parameters=None, status="Active"):
    """Add a new Indian market strategy"""
    return add_indian_strategy(name, description, symbol, strategy_type, parameters, status)

def update_strategy_indian(strategy_id, **kwargs):
    """Update an existing Indian market strategy"""
    return update_indian_strategy(strategy_id, **kwargs)

def delete_strategy_indian(strategy_id):
    """Delete an Indian market strategy"""
    return delete_indian_strategy(strategy_id)

def get_indian_data_sources():
    """Get a list of available Indian market data sources"""
    return get_indian_market_data_sources()