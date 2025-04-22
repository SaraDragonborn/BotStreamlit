"""
Alternative credentials management module that doesn't depend on secrets.toml
"""

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

def get_api_keys():
    """
    Gets API keys from session state, environment variables, or user input.
    Returns a tuple of (alpaca_api_key, alpaca_api_secret, has_keys)
    """
    # Check if keys are in session state
    if 'alpaca_api_key' in st.session_state and 'alpaca_api_secret' in st.session_state:
        return st.session_state.alpaca_api_key, st.session_state.alpaca_api_secret, True

    # Check environment variables
    alpaca_api_key = os.environ.get('ALPACA_API_KEY')
    alpaca_api_secret = os.environ.get('ALPACA_API_SECRET')
    
    # If both are available, save to session state and return
    if alpaca_api_key and alpaca_api_secret:
        st.session_state.alpaca_api_key = alpaca_api_key
        st.session_state.alpaca_api_secret = alpaca_api_secret
        return alpaca_api_key, alpaca_api_secret, True
    
    # Otherwise return empty values
    return None, None, False

def save_api_keys(alpaca_api_key, alpaca_api_secret):
    """Save API keys to session state"""
    st.session_state.alpaca_api_key = alpaca_api_key
    st.session_state.alpaca_api_secret = alpaca_api_secret
    return True

def api_keys_ui():
    """Display UI for entering API keys"""
    with st.sidebar.expander("API Keys", expanded=not has_keys()):
        alpaca_api_key = st.text_input("Alpaca API Key", 
                                       value=st.session_state.get('alpaca_api_key', ''),
                                       type="password")
        alpaca_api_secret = st.text_input("Alpaca API Secret", 
                                         value=st.session_state.get('alpaca_api_secret', ''),
                                         type="password")
        
        if st.button("Save API Keys"):
            save_api_keys(alpaca_api_key, alpaca_api_secret)
            st.success("API keys saved!")
            st.rerun()

def has_keys():
    """Check if API keys are available"""
    return 'alpaca_api_key' in st.session_state and st.session_state.alpaca_api_key != ''