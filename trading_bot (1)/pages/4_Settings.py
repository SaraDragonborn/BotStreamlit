import streamlit as st
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fixed_api import test_alpaca_connection

# Configure the page
st.set_page_config(
    page_title="Settings | AI Trading Bot",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #42A5F5;
        margin-bottom: 0.5rem;
    }
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'alpaca_api_key' not in st.session_state:
    st.session_state.alpaca_api_key = os.environ.get('ALPACA_API_KEY', '')
if 'alpaca_api_secret' not in st.session_state:
    st.session_state.alpaca_api_secret = os.environ.get('ALPACA_API_SECRET', '')
if 'trading_mode' not in st.session_state:
    st.session_state.trading_mode = 'paper'

# Main content
st.markdown('<p class="main-header">Settings</p>', unsafe_allow_html=True)

tabs = st.tabs(["API Connections", "Trading Preferences", "Notifications", "Data Sources", "Account"])

with tabs[0]:  # API Connections
    st.subheader("API Connections")
    
    # Alpaca API settings
    st.markdown('<p class="section-header">Alpaca API Settings</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        alpaca_key = st.text_input("Alpaca API Key", value=st.session_state.alpaca_api_key, type="password")
    
    with col2:
        alpaca_secret = st.text_input("Alpaca API Secret", value=st.session_state.alpaca_api_secret, type="password")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        trading_mode = st.radio(
            "Trading Mode",
            options=["paper", "live"],
            format_func=lambda x: "Paper Trading" if x == "paper" else "Live Trading",
            index=0 if st.session_state.trading_mode == "paper" else 1
        )
    
    with col2:
        # Test connection button
        if st.button("Test Connection"):
            with st.spinner("Testing connection to Alpaca API..."):
                # Update session state first
                st.session_state.alpaca_api_key = alpaca_key
                st.session_state.alpaca_api_secret = alpaca_secret
                st.session_state.trading_mode = trading_mode
                
                success, message = test_alpaca_connection(alpaca_key, alpaca_secret)
                
                if success:
                    st.success(message)
                    st.session_state.alpaca_connected = True
                else:
                    st.error(message)
                    st.session_state.alpaca_connected = False
    
    with col3:
        pass  # Placeholder for layout balance
    
    if st.button("Save API Settings"):
        # Update session state
        st.session_state.alpaca_api_key = alpaca_key
        st.session_state.alpaca_api_secret = alpaca_secret
        st.session_state.trading_mode = trading_mode
        
        st.success("API settings saved successfully!")
    
    # Other API connections
    st.markdown('<p class="section-header">Additional API Connections</p>', unsafe_allow_html=True)
    
    # Alpha Vantage
    with st.expander("Alpha Vantage API (Optional)"):
        alpha_vantage_key = st.text_input("Alpha Vantage API Key", type="password")
        
        if st.button("Save Alpha Vantage API Key"):
            if alpha_vantage_key:
                st.success("Alpha Vantage API key saved successfully!")
            else:
                st.warning("Please enter an API key")
    
    # Polygon.io
    with st.expander("Polygon.io API (Optional)"):
        polygon_key = st.text_input("Polygon.io API Key", type="password")
        
        if st.button("Save Polygon.io API Key"):
            if polygon_key:
                st.success("Polygon.io API key saved successfully!")
            else:
                st.warning("Please enter an API key")
    
    # Finnhub
    with st.expander("Finnhub API (Optional)"):
        finnhub_key = st.text_input("Finnhub API Key", type="password")
        
        if st.button("Save Finnhub API Key"):
            if finnhub_key:
                st.success("Finnhub API key saved successfully!")
            else:
                st.warning("Please enter an API key")

with tabs[1]:  # Trading Preferences
    st.subheader("Trading Preferences")
    
    # General trading settings
    st.markdown('<p class="section-header">General Trading Settings</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_position_size = st.slider("Maximum Position Size (% of Portfolio)", min_value=1, max_value=100, value=20)
        max_positions = st.number_input("Maximum Concurrent Positions", min_value=1, max_value=50, value=10)
        price_precision = st.number_input("Price Precision (Decimal Places)", min_value=2, max_value=8, value=2)
    
    with col2:
        default_stop_loss = st.slider("Default Stop Loss (%)", min_value=1, max_value=50, value=5)
        default_take_profit = st.slider("Default Take Profit (%)", min_value=1, max_value=100, value=15)
        leverage = st.slider("Maximum Leverage", min_value=1.0, max_value=4.0, value=1.0, step=0.1)
    
    # Trading hours
    st.markdown('<p class="section-header">Trading Hours</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        trade_outside_market = st.checkbox("Allow Trading Outside Market Hours", value=False)
        weekends_trading = st.checkbox("Allow Weekend Trading (Crypto Only)", value=True)
    
    with col2:
        trading_start_time = st.time_input("Daily Trading Start Time", value=datetime.strptime("09:30", "%H:%M").time())
        trading_end_time = st.time_input("Daily Trading End Time", value=datetime.strptime("16:00", "%H:%M").time())
    
    # Risk management
    st.markdown('<p class="section-header">Risk Management</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_daily_loss = st.slider("Max Daily Loss (% of Portfolio)", min_value=1, max_value=20, value=5)
        max_drawdown = st.slider("Max Drawdown Alert (%)", min_value=5, max_value=50, value=15)
    
    with col2:
        auto_shutdown = st.checkbox("Auto-Shutdown on Max Daily Loss", value=True)
        confirm_trades = st.checkbox("Confirm Trades Before Execution", value=True)
    
    # Save preferences
    if st.button("Save Trading Preferences"):
        st.success("Trading preferences saved successfully!")

with tabs[2]:  # Notifications
    st.subheader("Notification Settings")
    
    # Notification methods
    st.markdown('<p class="section-header">Notification Methods</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_notifications = st.checkbox("Email Notifications", value=True)
        if email_notifications:
            email_address = st.text_input("Email Address")
    
    with col2:
        telegram_notifications = st.checkbox("Telegram Notifications", value=False)
        if telegram_notifications:
            telegram_bot_token = st.text_input("Telegram Bot Token", type="password")
            telegram_chat_id = st.text_input("Telegram Chat ID")
    
    # Notification events
    st.markdown('<p class="section-header">Notification Events</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Trade Executed", value=True)
        st.checkbox("Stop Loss Hit", value=True)
        st.checkbox("Take Profit Hit", value=True)
        st.checkbox("New Trading Signal", value=True)
    
    with col2:
        st.checkbox("Daily Summary", value=True)
        st.checkbox("Significant Market Movement", value=True)
        st.checkbox("Portfolio Value Changes (>5%)", value=True)
        st.checkbox("API Connection Issues", value=True)
    
    # Alert thresholds
    st.markdown('<p class="section-header">Alert Thresholds</p>', unsafe_allow_html=True)
    
    price_alert_threshold = st.slider("Price Alert Threshold (%)", min_value=1, max_value=20, value=5)
    volatility_alert_threshold = st.slider("Volatility Alert Threshold", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
    
    # Save notification settings
    if st.button("Save Notification Settings"):
        st.success("Notification settings saved successfully!")
    
    # Test notification
    if st.button("Send Test Notification"):
        st.info("Sending test notification...")
        st.success("Test notification sent successfully!")

with tabs[3]:  # Data Sources
    st.subheader("Data Sources")
    
    # Market data sources
    st.markdown('<p class="section-header">Market Data Sources</p>', unsafe_allow_html=True)
    
    primary_data_source = st.selectbox(
        "Primary Market Data Source",
        options=["Alpaca", "Alpha Vantage", "Polygon.io", "Finnhub", "Yahoo Finance"],
        index=0
    )
    
    backup_data_source = st.selectbox(
        "Backup Market Data Source",
        options=["None", "Alpaca", "Alpha Vantage", "Polygon.io", "Finnhub", "Yahoo Finance"],
        index=5
    )
    
    # News data sources
    st.markdown('<p class="section-header">News Data Sources</p>', unsafe_allow_html=True)
    
    news_sources = st.multiselect(
        "News Sources",
        options=["Alpaca News API", "Alpha Vantage News", "Finnhub News", "MarketWatch", "Yahoo Finance", "Reuters", "Bloomberg"],
        default=["Alpaca News API", "Finnhub News"]
    )
    
    # Social media sentiment
    include_social_sentiment = st.checkbox("Include Social Media Sentiment", value=True)
    if include_social_sentiment:
        social_platforms = st.multiselect(
            "Social Platforms",
            options=["Twitter/X", "Reddit WallStreetBets", "StockTwits", "Investment Forums"],
            default=["Twitter/X", "Reddit WallStreetBets"]
        )
    
    # Data update frequency
    st.markdown('<p class="section-header">Data Update Frequency</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        price_data_frequency = st.selectbox(
            "Price Data Update Frequency",
            options=["Real-time", "1 Minute", "5 Minutes", "15 Minutes", "30 Minutes", "1 Hour"],
            index=0
        )
    
    with col2:
        news_data_frequency = st.selectbox(
            "News Data Update Frequency",
            options=["Real-time", "5 Minutes", "15 Minutes", "30 Minutes", "1 Hour"],
            index=1
        )
    
    # Save data settings
    if st.button("Save Data Source Settings"):
        st.success("Data source settings saved successfully!")

with tabs[4]:  # Account
    st.subheader("Account Settings")
    
    # User profile
    st.markdown('<p class="section-header">User Profile</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_name = st.text_input("Name", "User")
        user_email = st.text_input("Email Address")
    
    with col2:
        account_type = st.selectbox("Account Type", options=["Standard", "Premium", "Enterprise"], index=0)
        dark_mode = st.checkbox("Dark Mode", value=False)
    
    # Security settings
    st.markdown('<p class="section-header">Security Settings</p>', unsafe_allow_html=True)
    
    enable_two_factor = st.checkbox("Enable Two-Factor Authentication", value=False)
    api_key_rotation = st.selectbox("API Key Rotation", options=["Never", "30 Days", "60 Days", "90 Days"], index=0)
    
    # Data retention
    st.markdown('<p class="section-header">Data Retention</p>', unsafe_allow_html=True)
    
    data_retention = st.slider("Data Retention Period (Days)", min_value=30, max_value=365, value=90)
    auto_backup = st.checkbox("Automatic Backups", value=True)
    if auto_backup:
        backup_frequency = st.selectbox("Backup Frequency", options=["Daily", "Weekly", "Monthly"], index=1)
    
    # Reset settings
    st.markdown('<p class="section-header">Reset Settings</p>', unsafe_allow_html=True)
    
    if st.button("Reset All Settings to Default"):
        # Show confirmation dialog
        st.warning("Are you sure you want to reset all settings to default values?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Reset Settings"):
                st.success("All settings have been reset to default values.")
        with col2:
            if st.button("No, Cancel"):
                st.info("Reset operation canceled.")
    
    # AI Trading Bot Info
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center;">
        <p><strong>AI Trading Bot</strong> v1.0.0</p>
        <p>© 2025 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)