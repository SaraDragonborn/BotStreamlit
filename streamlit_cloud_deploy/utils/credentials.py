import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def get_alpaca_credentials():
    """Get Alpaca API credentials from Streamlit secrets or environment"""
    # Check for credentials in Streamlit secrets
    if 'alpaca' in st.secrets:
        return {
            'api_key': st.secrets.alpaca.api_key,
            'api_secret': st.secrets.alpaca.api_secret,
            'paper_trading': st.secrets.alpaca.get('paper_trading', True)
        }
    # Check for credentials in environment variables
    elif 'ALPACA_API_KEY' in os.environ and 'ALPACA_API_SECRET' in os.environ:
        return {
            'api_key': os.environ['ALPACA_API_KEY'],
            'api_secret': os.environ['ALPACA_API_SECRET'],
            'paper_trading': os.environ.get('ALPACA_PAPER_TRADING', 'true').lower() == 'true'
        }
    # Check for credentials in session state (set through the UI)
    elif 'alpaca_api_key' in st.session_state and 'alpaca_api_secret' in st.session_state:
        return {
            'api_key': st.session_state.alpaca_api_key,
            'api_secret': st.session_state.alpaca_api_secret,
            'paper_trading': st.session_state.get('alpaca_paper_trading', True)
        }
    # No credentials found
    return None

def get_angel_one_credentials():
    """Get Angel One API credentials from Streamlit secrets or environment"""
    # Check for credentials in Streamlit secrets
    if 'angel_one' in st.secrets:
        return {
            'api_key': st.secrets.angel_one.api_key,
            'client_id': st.secrets.angel_one.get('client_id', ''),
            'password': st.secrets.angel_one.get('password', ''),
            'market_feed_api_key': st.secrets.angel_one.get('market_feed_api_key', ''),
            'historical_api_key': st.secrets.angel_one.get('historical_api_key', '')
        }
    # Check for credentials in environment variables
    elif 'ANGEL_ONE_API_KEY' in os.environ:
        return {
            'api_key': os.environ['ANGEL_ONE_API_KEY'],
            'client_id': os.environ.get('ANGEL_ONE_CLIENT_ID', ''),
            'password': os.environ.get('ANGEL_ONE_PASSWORD', ''),
            'market_feed_api_key': os.environ.get('ANGEL_ONE_MARKET_FEED_API_KEY', ''),
            'historical_api_key': os.environ.get('ANGEL_ONE_HISTORICAL_API_KEY', '')
        }
    # Check for credentials in session state (set through the UI)
    elif 'angel_one_api_key' in st.session_state:
        return {
            'api_key': st.session_state.angel_one_api_key,
            'client_id': st.session_state.get('angel_one_client_id', ''),
            'password': st.session_state.get('angel_one_password', ''),
            'market_feed_api_key': st.session_state.get('angel_one_market_feed_api_key', ''),
            'historical_api_key': st.session_state.get('angel_one_historical_api_key', '')
        }
    # No credentials found
    return None

def set_credentials_in_session_state(api_name, **credentials):
    """Store API credentials in session state"""
    if api_name == 'alpaca':
        st.session_state.alpaca_api_key = credentials.get('api_key', '')
        st.session_state.alpaca_api_secret = credentials.get('api_secret', '')
        st.session_state.alpaca_paper_trading = credentials.get('paper_trading', True)
    elif api_name == 'angel_one':
        st.session_state.angel_one_api_key = credentials.get('api_key', '')
        st.session_state.angel_one_client_id = credentials.get('client_id', '')
        st.session_state.angel_one_password = credentials.get('password', '')
        st.session_state.angel_one_market_feed_api_key = credentials.get('market_feed_api_key', '')
        st.session_state.angel_one_historical_api_key = credentials.get('historical_api_key', '')
    return True