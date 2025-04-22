import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add utils directory to path if not already added
if os.path.abspath("utils") not in sys.path:
    sys.path.append(os.path.abspath("utils"))

# Import utility modules
from utils.api import (
    get_account_info, get_positions, get_orders, 
    get_asset_price, get_fingpt_news, get_fingpt_signals,
    test_alpaca_connection
)
from utils.charts import (
    format_currency, format_percentage, plot_portfolio_composition
)
from utils.credentials import load_credentials, save_credentials
from utils.strategies import load_strategies

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
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
    }
    .positive {
        color: #4CAF50;
    }
    .negative {
        color: #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'alpaca_api_key' not in st.session_state:
    st.session_state.alpaca_api_key = ''
if 'alpaca_api_secret' not in st.session_state:
    st.session_state.alpaca_api_secret = ''
if 'trading_mode' not in st.session_state:
    st.session_state.trading_mode = 'paper'
if 'alpaca_connected' not in st.session_state:
    st.session_state.alpaca_connected = False
if 'strategies' not in st.session_state:
    st.session_state.strategies = load_strategies()

# Load credentials
credentials = load_credentials()
alpaca_api_key = credentials.get('alpaca_api_key', '')
alpaca_api_secret = credentials.get('alpaca_api_secret', '')
trading_mode = credentials.get('trading_mode', 'paper')

# Update session state with loaded credentials
if alpaca_api_key:
    st.session_state.alpaca_api_key = alpaca_api_key
if alpaca_api_secret:
    st.session_state.alpaca_api_secret = alpaca_api_secret
if trading_mode:
    st.session_state.trading_mode = trading_mode

# Check if we have API keys in environment variables
if not st.session_state.alpaca_api_key and os.environ.get('ALPACA_API_KEY'):
    st.session_state.alpaca_api_key = os.environ.get('ALPACA_API_KEY')
if not st.session_state.alpaca_api_secret and os.environ.get('ALPACA_API_SECRET'):
    st.session_state.alpaca_api_secret = os.environ.get('ALPACA_API_SECRET')

# Sidebar - API Connection and Settings
with st.sidebar:
    st.image("https://www.svgrepo.com/show/249756/stock-market-investment.svg", width=100)
    st.markdown("## AI Trading Bot")
    
    st.markdown("### Connection Settings")
    alpaca_key = st.text_input("Alpaca API Key", value=st.session_state.alpaca_api_key, type="password")
    alpaca_secret = st.text_input("Alpaca API Secret", value=st.session_state.alpaca_api_secret, type="password")
    
    trading_mode = st.radio(
        "Trading Mode",
        options=["paper", "live"],
        format_func=lambda x: "Paper Trading" if x == "paper" else "Live Trading",
        index=0 if st.session_state.trading_mode == "paper" else 1
    )
    
    # Update session state if values changed
    if alpaca_key != st.session_state.alpaca_api_key:
        st.session_state.alpaca_api_key = alpaca_key
    if alpaca_secret != st.session_state.alpaca_api_secret:
        st.session_state.alpaca_api_secret = alpaca_secret
    if trading_mode != st.session_state.trading_mode:
        st.session_state.trading_mode = trading_mode
    
    # Save credentials button
    if st.button("Save Credentials"):
        success, message = save_credentials(alpaca_key, alpaca_secret, trading_mode)
        if success:
            st.success(message)
        else:
            st.info(message)
    
    # Connection status and test button
    col1, col2 = st.columns(2)
    
    with col1:
        connection_status = "🟢 Connected" if st.session_state.alpaca_connected else "🔴 Disconnected"
        st.markdown(f"**Status**: {connection_status}")
    
    with col2:
        if st.button("Test Connection"):
            with st.spinner("Testing connection..."):
                success, message = test_alpaca_connection(
                    st.session_state.alpaca_api_key,
                    st.session_state.alpaca_api_secret
                )
                if success:
                    st.success(message)
                    st.session_state.alpaca_connected = True
                else:
                    st.error(message)
                    st.session_state.alpaca_connected = False
    
    st.markdown("---")
    
    # Quick links
    st.markdown("### Quick Links")
    st.markdown("[📈 Portfolio](Portfolio)")
    st.markdown("[🧠 Strategies](Strategies)")
    st.markdown("[🤖 AI Analysis](AI_Analysis)")
    st.markdown("[⚙️ Settings](Settings)")
    
    st.markdown("---")
    st.markdown("© 2025 AI Trading Bot")

# Main Dashboard
st.markdown('<p class="main-header">Trading Dashboard</p>', unsafe_allow_html=True)

# Check if connected
if not st.session_state.alpaca_connected:
    # Try to connect automatically if we have credentials
    if st.session_state.alpaca_api_key and st.session_state.alpaca_api_secret:
        with st.spinner("Connecting to Alpaca API..."):
            success, message = test_alpaca_connection(
                st.session_state.alpaca_api_key,
                st.session_state.alpaca_api_secret
            )
            if success:
                st.session_state.alpaca_connected = True

# Main content
if st.session_state.alpaca_connected:
    # Get account info
    with st.spinner("Loading account information..."):
        account = get_account_info()
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if account:
            equity = float(account.get('equity', 0))
            st.markdown(f'<p class="metric-label">Portfolio Value</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{format_currency(equity)}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="metric-label">Portfolio Value</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">$0.00</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if account:
            cash = float(account.get('cash', 0))
            st.markdown(f'<p class="metric-label">Available Cash</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{format_currency(cash)}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="metric-label">Available Cash</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">$0.00</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if account:
            pnl = float(account.get('equity', 0)) - float(account.get('last_equity', account.get('equity', 0)))
            pnl_percent = (pnl / float(account.get('last_equity', account.get('equity', 1)))) * 100 if float(account.get('last_equity', 0)) > 0 else 0
            st.markdown(f'<p class="metric-label">Today\'s P&L</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{format_currency(pnl)} <small>{format_percentage(pnl_percent)}</small></p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="metric-label">Today\'s P&L</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">$0.00 <small>+0.00%</small></p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.spinner("Loading positions..."):
            positions = get_positions()
        num_positions = len(positions)
        st.markdown(f'<p class="metric-label">Active Positions</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{num_positions}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Portfolio and performance row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">Portfolio Composition</p>', unsafe_allow_html=True)
        
        if positions:
            # Create portfolio pie chart using our utility function
            fig = plot_portfolio_composition(positions, account)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No positions in portfolio")
            
            # Show suggested positions based on strategies
            active_strategies = [s for s in st.session_state.strategies if s["status"] == "Active"]
            if active_strategies:
                st.markdown("### Suggested Positions")
                st.markdown("Based on your active strategies, the following positions are recommended:")
                
                for strategy in active_strategies[:2]:  # Show top 2 strategies
                    for symbol in strategy["symbols"][:2]:  # Show top 2 symbols per strategy
                        col1, col2, col3 = st.columns([2, 2, 3])
                        with col1:
                            st.markdown(f"**{symbol}**")
                        with col2:
                            st.markdown(f"{strategy['name']}")
                        with col3:
                            if st.button(f"View Details for {symbol}", key=f"view_{symbol}"):
                                st.session_state.selected_symbol = symbol
                                st.rerun()
    
    with col2:
        st.markdown('<p class="sub-header">Recent Signals</p>', unsafe_allow_html=True)
        
        # Sample tickers to show signals for - in a real app, this would be dynamically generated
        # from active strategies or top holdings
        tickers = ["AAPL", "MSFT", "AMZN"]
        signals_data = []
        
        with st.spinner("Fetching trading signals..."):
            for ticker in tickers:
                signal = get_fingpt_signals(ticker)
                if signal:
                    signals_data.append({
                        'Ticker': ticker,
                        'Signal': signal.get('recommendation', 'NEUTRAL'),
                        'Strength': signal.get('signal_strength', 0),
                        'Current Price': format_currency(signal.get('current_price', 0)),
                        'Date': signal.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))
                    })
        
        if signals_data:
            signals_df = pd.DataFrame(signals_data)
            st.dataframe(signals_df, use_container_width=True, hide_index=True)
            
            # Add a "View Analysis" button
            if st.button("View Detailed Analysis"):
                # This would navigate to the AI Analysis page in a real app
                st.info("Navigate to the AI Analysis page for detailed signal analysis")
        else:
            st.info("No recent signals available")
            
            # Add a "Generate Signals" button
            if st.button("Generate New Signals"):
                st.info("Navigating to AI Analysis page to generate new signals")
    
    # Recent activity and news row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">Recent Orders</p>', unsafe_allow_html=True)
        
        with st.spinner("Fetching recent orders..."):
            orders = get_orders()
        
        if orders:
            # Convert to DataFrame for display
            orders_data = []
            for order in orders[:5]:  # Show only 5 most recent
                orders_data.append({
                    'Symbol': order.get('symbol', ''),
                    'Side': order.get('side', '').capitalize(),
                    'Qty': order.get('qty', ''),
                    'Type': order.get('type', '').upper(),
                    'Status': order.get('status', '').capitalize(),
                    'Submitted At': order.get('submitted_at', '')[:19].replace('T', ' ')  # Format datetime
                })
            
            orders_df = pd.DataFrame(orders_data)
            st.dataframe(orders_df, use_container_width=True, hide_index=True)
            
            # Add "View All Orders" button
            if st.button("View All Orders"):
                st.info("Navigate to the Portfolio page to see all orders")
        else:
            st.info("No recent orders")
            
            # Add "Place Order" button
            if st.button("Place New Order"):
                st.info("Navigate to the Portfolio page to place a new order")
    
    with col2:
        st.markdown('<p class="sub-header">Market News</p>', unsafe_allow_html=True)
        
        # Get news for SPY as a market proxy
        with st.spinner("Fetching market news..."):
            news = get_fingpt_news("SPY", days=3, limit=3)
        
        if news:
            for item in news:
                title = item.get('title', 'No title')
                date = item.get('date', '')
                source = item.get('source', 'Unknown')
                url = item.get('url', '#')
                
                st.markdown(f"""
                <div style="border-bottom: 1px solid #ddd; padding-bottom: 10px; margin-bottom: 10px;">
                    <h4 style="margin-bottom: 5px;">{title}</h4>
                    <p style="color: #666; font-size: 0.8rem;">{date} • {source}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add "View More News" button
            if st.button("View More News"):
                st.info("Navigate to the AI Analysis page to see more news and sentiment analysis")
        else:
            st.info("No recent news available")
            
            # Add "Fetch News" button
            if st.button("Fetch Latest News"):
                st.info("Navigate to the AI Analysis page to fetch the latest market news")
    
    # Active strategies section
    st.markdown('<p class="sub-header">Active Trading Strategies</p>', unsafe_allow_html=True)
    
    # Filter active strategies
    active_strategies = [s for s in st.session_state.strategies if s["status"] == "Active"]
    
    if active_strategies:
        # Create columns for each strategy (up to 3)
        cols = st.columns(min(3, len(active_strategies)))
        
        for i, strategy in enumerate(active_strategies[:3]):  # Show up to 3 active strategies
            with cols[i]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"### {strategy['name']}")
                st.markdown(f"*{strategy['description']}*")
                st.markdown(f"**Symbols:** {', '.join(strategy['symbols'])}")
                
                # Performance metrics
                performance = strategy.get('performance', {})
                col1, col2 = st.columns(2)
                col1.metric("Win Rate", f"{performance.get('win_rate', 0)}%")
                col2.metric("Profit Factor", f"{performance.get('profit_factor', 0)}")
                
                if st.button("View Details", key=f"strategy_{strategy['id']}"):
                    # In a real app, this would navigate to the strategy detail page
                    st.info(f"View detailed information for {strategy['name']} in the Strategies page")
                
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No active strategies. Go to the Strategies page to create and activate strategies.")
        
        if st.button("Create New Strategy"):
            # In a real app, this would navigate to the strategy creation page
            st.info("Navigate to the Strategies page to create a new trading strategy")

else:
    # Not connected - show onboarding information
    st.warning("⚠️ Please connect to Alpaca API to access all trading functionality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Getting Started")
        st.markdown("""
        1. Enter your Alpaca API credentials in the sidebar
        2. Choose between paper trading (simulated) or live trading
        3. Click "Test Connection" to verify your credentials
        4. Explore trading strategies and AI-powered analysis
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Don't have Alpaca API keys?")
        st.markdown("""
        1. Sign up for an account at [Alpaca](https://app.alpaca.markets/signup)
        2. Create API keys in your dashboard
        3. For testing, use paper trading keys
        4. Enter the keys in the sidebar to connect
        """)
        st.markdown("""
        [Sign up for Alpaca](https://app.alpaca.markets/signup)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Features preview
    st.markdown("### AI Trading Bot Features")
    
    feature_cols = st.columns(3)
    
    with feature_cols[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Trading Strategies")
        st.markdown("""
        - Create custom trading strategies
        - Backtest with historical data
        - Technical indicator strategies
        - AI-powered algorithms
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with feature_cols[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Portfolio Management")
        st.markdown("""
        - Track all positions and orders
        - Analyze portfolio performance
        - Place orders through the UI
        - Monitor P&L and metrics
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with feature_cols[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### AI Analysis")
        st.markdown("""
        - News sentiment analysis
        - Trading signal generation
        - Market trend predictions
        - Custom ML model training
        """)
        st.markdown('</div>', unsafe_allow_html=True)