import os
import json
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def save_credentials(alpaca_api_key, alpaca_api_secret, trading_mode="paper"):
    """
    Save API credentials to session state and environment variables
    """
    # Update session state
    st.session_state.alpaca_api_key = alpaca_api_key
    st.session_state.alpaca_api_secret = alpaca_api_secret
    st.session_state.trading_mode = trading_mode
    
    # Save to .streamlit/secrets.toml if we're in a local environment
    try:
        os.makedirs(".streamlit", exist_ok=True)
        
        # Create or update secrets.toml
        with open(".streamlit/secrets.toml", "w") as f:
            f.write(f"ALPACA_API_KEY = \"{alpaca_api_key}\"\n")
            f.write(f"ALPACA_API_SECRET = \"{alpaca_api_secret}\"\n")
            f.write(f"TRADING_MODE = \"{trading_mode}\"\n")
        
        return True, "Credentials saved successfully"
    except Exception as e:
        # If we can't write to the file, it might be a read-only environment like Streamlit Cloud
        # In that case, credentials are already stored in Streamlit secrets
        return False, f"Could not save credentials to file: {str(e)}"

def load_credentials():
    """
    Load API credentials from environment or Streamlit secrets
    """
    # Priority: 
    # 1. Session state (already loaded)
    # 2. Streamlit secrets
    # 3. Environment variables
    
    # Check if we already have credentials in session state
    if 'alpaca_api_key' in st.session_state and st.session_state.alpaca_api_key:
        return {
            'alpaca_api_key': st.session_state.alpaca_api_key,
            'alpaca_api_secret': st.session_state.alpaca_api_secret,
            'trading_mode': st.session_state.trading_mode if 'trading_mode' in st.session_state else 'paper'
        }
    
    # Try to load from Streamlit secrets
    try:
        alpaca_api_key = st.secrets["ALPACA_API_KEY"]
        alpaca_api_secret = st.secrets["ALPACA_API_SECRET"]
        trading_mode = st.secrets.get("TRADING_MODE", "paper")
        
        # Update session state
        st.session_state.alpaca_api_key = alpaca_api_key
        st.session_state.alpaca_api_secret = alpaca_api_secret
        st.session_state.trading_mode = trading_mode
        
        return {
            'alpaca_api_key': alpaca_api_key,
            'alpaca_api_secret': alpaca_api_secret,
            'trading_mode': trading_mode
        }
    except Exception:
        # If no streamlit secrets, try environment variables
        alpaca_api_key = os.environ.get('ALPACA_API_KEY', '')
        alpaca_api_secret = os.environ.get('ALPACA_API_SECRET', '')
        trading_mode = os.environ.get('TRADING_MODE', 'paper')
        
        # Update session state
        st.session_state.alpaca_api_key = alpaca_api_key
        st.session_state.alpaca_api_secret = alpaca_api_secret
        st.session_state.trading_mode = trading_mode
        
        return {
            'alpaca_api_key': alpaca_api_key,
            'alpaca_api_secret': alpaca_api_secret,
            'trading_mode': trading_mode
        }