"""
Streamlit App for Multi-Asset Trading Bot
=======================================
User interface for the trading bot using Streamlit.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any

# Set page config
st.set_page_config(
    page_title="Multi-Asset Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "performance"), exist_ok=True)
os.makedirs(os.path.join(LOGS_DIR, "trades"), exist_ok=True)

# Load configuration
def load_config():
    """Load configuration from config.py"""
    try:
        from config import get_config
        return get_config()
    except ImportError:
        st.error("Configuration module not found. Make sure config.py exists.")
        return {}

# Global state
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False
if 'trading_bot' not in st.session_state:
    st.session_state.trading_bot = None
if 'market_status' not in st.session_state:
    st.session_state.market_status = {}
if 'portfolio_value' not in st.session_state:
    st.session_state.portfolio_value = 0
if 'buying_power' not in st.session_state:
    st.session_state.buying_power = 0
if 'positions' not in st.session_state:
    st.session_state.positions = {}

# Sidebar
st.sidebar.title("Trading Bot Control")

# API connection
st.sidebar.header("API Connection")

# US Market (Alpaca)
st.sidebar.subheader("Alpaca API (US Markets)")
alpaca_api_key = st.sidebar.text_input("Alpaca API Key", type="password")
alpaca_api_secret = st.sidebar.text_input("Alpaca API Secret", type="password")
alpaca_paper = st.sidebar.checkbox("Use Paper Trading", value=True)

# Indian Market (Angel One)
st.sidebar.subheader("Angel One API (Indian Markets)")
angel_api_key = st.sidebar.text_input("Angel API Key", type="password")
angel_client_id = st.sidebar.text_input("Angel Client ID")
angel_password = st.sidebar.text_input("Angel Password", type="password")

# Market selection
st.sidebar.header("Markets")
us_market = st.sidebar.checkbox("US Market", value=True)
india_market = st.sidebar.checkbox("Indian Market")
crypto_market = st.sidebar.checkbox("Crypto Market")
forex_market = st.sidebar.checkbox("Forex Market")

# Strategy selection
st.sidebar.header("Strategy")
strategy_mode = st.sidebar.selectbox(
    "Strategy Selection Mode",
    options=["adaptive", "performance", "rotation", "ensemble"],
    index=0
)

# Bot control
st.sidebar.header("Control")
start_button = st.sidebar.button("Run Trading Cycle")
schedule_button = st.sidebar.button("Schedule Trading")
stop_button = st.sidebar.button("Stop Bot")

# Main content
st.title("Multi-Asset Trading Bot Dashboard")

# Market Status
st.header("Market Status")
market_status_cols = st.columns(4)

# Load custom CSS
st.markdown("""
<style>
    .market-status {
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .market-open {
        background-color: rgba(0, 255, 0, 0.2);
    }
    .market-closed {
        background-color: rgba(255, 0, 0, 0.2);
    }
    .metric-card {
        border: 1px solid #f0f0f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .position-row {
        margin-bottom: 10px;
        padding: 10px;
        border-radius: 5px;
    }
    .profit {
        background-color: rgba(0, 255, 0, 0.1);
    }
    .loss {
        background-color: rgba(255, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Function to check if markets are open
def check_markets():
    """Check if markets are open (simplified version)"""
    markets = {}
    
    # US Market
    now = datetime.datetime.now()
    is_weekday = now.weekday() < 5  # Monday to Friday
    is_market_hours = 9 <= now.hour < 16  # 9:00 AM to 4:00 PM
    markets['US'] = is_weekday and is_market_hours
    
    # Indian Market
    india_time = now + datetime.timedelta(hours=5, minutes=30)  # Crude IST conversion
    is_weekday = india_time.weekday() < 5  # Monday to Friday
    is_market_hours = (
        (india_time.hour > 9 or (india_time.hour == 9 and india_time.minute >= 15)) and
        (india_time.hour < 15 or (india_time.hour == 15 and india_time.minute <= 30))
    )
    markets['India'] = is_weekday and is_market_hours
    
    # Crypto Market (always open)
    markets['Crypto'] = True
    
    # Forex Market (open 24/5)
    markets['Forex'] = is_weekday
    
    return markets

# Display market status
market_status = check_markets()
st.session_state.market_status = market_status

with market_status_cols[0]:
    status = "OPEN" if market_status['US'] else "CLOSED"
    status_class = "market-open" if market_status['US'] else "market-closed"
    st.markdown(f"""
    <div class="market-status {status_class}">
        US Market: {status}
    </div>
    """, unsafe_allow_html=True)

with market_status_cols[1]:
    status = "OPEN" if market_status['India'] else "CLOSED"
    status_class = "market-open" if market_status['India'] else "market-closed"
    st.markdown(f"""
    <div class="market-status {status_class}">
        Indian Market: {status}
    </div>
    """, unsafe_allow_html=True)

with market_status_cols[2]:
    status = "OPEN" if market_status['Crypto'] else "CLOSED"
    status_class = "market-open" if market_status['Crypto'] else "market-closed"
    st.markdown(f"""
    <div class="market-status {status_class}">
        Crypto Market: {status}
    </div>
    """, unsafe_allow_html=True)

with market_status_cols[3]:
    status = "OPEN" if market_status['Forex'] else "CLOSED"
    status_class = "market-open" if market_status['Forex'] else "market-closed"
    st.markdown(f"""
    <div class="market-status {status_class}">
        Forex Market: {status}
    </div>
    """, unsafe_allow_html=True)

# Portfolio Overview
st.header("Portfolio Overview")
portfolio_cols = st.columns(3)

with portfolio_cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Portfolio Value</h3>
        <h1>${st.session_state.portfolio_value:,.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

with portfolio_cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Buying Power</h3>
        <h1>${st.session_state.buying_power:,.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

with portfolio_cols[2]:
    position_count = len(st.session_state.positions)
    st.markdown(f"""
    <div class="metric-card">
        <h3>Active Positions</h3>
        <h1>{position_count}</h1>
    </div>
    """, unsafe_allow_html=True)

# Performance Metrics
st.header("Performance Metrics")
performance_cols = st.columns(4)

# Load performance data (simulated for now)
try:
    performance_file = os.path.join(DATA_DIR, "performance", "performance_summary.json")
    
    if os.path.exists(performance_file):
        with open(performance_file, 'r') as f:
            performance_data = json.load(f)
    else:
        performance_data = {
            'total_pnl': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0
        }
except Exception as e:
    st.error(f"Error loading performance data: {str(e)}")
    performance_data = {
        'total_pnl': 0,
        'win_rate': 0,
        'profit_factor': 0,
        'max_drawdown': 0
    }

with performance_cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total P&L</h3>
        <h1>${performance_data.get('total_pnl', 0):,.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

with performance_cols[1]:
    win_rate = performance_data.get('win_rate', 0) * 100 if isinstance(performance_data.get('win_rate', 0), float) else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3>Win Rate</h3>
        <h1>{win_rate:.1f}%</h1>
    </div>
    """, unsafe_allow_html=True)

with performance_cols[2]:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Profit Factor</h3>
        <h1>{performance_data.get('profit_factor', 0):.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

with performance_cols[3]:
    max_dd = performance_data.get('max_drawdown', 0) * 100 if isinstance(performance_data.get('max_drawdown', 0), float) else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3>Max Drawdown</h3>
        <h1>{max_dd:.1f}%</h1>
    </div>
    """, unsafe_allow_html=True)

# Positions
st.header("Current Positions")

# Display positions
if not st.session_state.positions:
    st.info("No open positions")
else:
    for symbol, position in st.session_state.positions.items():
        # Determine if in profit or loss
        pnl = position.get('unrealized_pl', 0)
        row_class = "profit" if pnl >= 0 else "loss"
        
        # Display position
        st.markdown(f"""
        <div class="position-row {row_class}">
            <strong>{symbol}</strong> ({position.get('market', 'Unknown')}) - 
            {position.get('quantity', 0)} shares @ ${position.get('entry_price', 0):.2f} - 
            Current: ${position.get('current_price', 0):.2f} - 
            P&L: ${pnl:.2f} ({position.get('unrealized_plpc', 0) * 100:.2f}%)
        </div>
        """, unsafe_allow_html=True)

# Recent Trades
st.header("Recent Trades")

# Load trades (simulated for now)
try:
    trades_file = os.path.join(DATA_DIR, "performance", "trades.json")
    
    if os.path.exists(trades_file):
        with open(trades_file, 'r') as f:
            trades_data = json.load(f)
            if trades_data:
                # Sort by exit date (most recent first)
                trades_data = sorted(trades_data, key=lambda x: x.get('exit_date', ''), reverse=True)
                # Limit to 10 most recent
                trades_data = trades_data[:10]
    else:
        trades_data = []
except Exception as e:
    st.error(f"Error loading trades data: {str(e)}")
    trades_data = []

if not trades_data:
    st.info("No recent trades")
else:
    trades_df = pd.DataFrame(trades_data)
    
    # Format columns
    if not trades_df.empty:
        # Make a copy to avoid modifying the original
        display_df = trades_df.copy()
        
        # Add column for profit/loss indicator
        if 'pnl' in display_df.columns:
            display_df['result'] = display_df['pnl'].apply(lambda x: '🟢 Profit' if x > 0 else '🔴 Loss')
        
        # Format numeric columns
        for col in ['entry_price', 'exit_price', 'pnl']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "")
        
        # Format percentage columns
        if 'pnl_pct' in display_df.columns:
            display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "")
        
        # Reorder and select columns
        display_cols = ['symbol', 'entry_date', 'exit_date', 'shares', 'entry_price', 
                       'exit_price', 'pnl', 'pnl_pct', 'result', 'strategy', 'market']
        display_cols = [col for col in display_cols if col in display_df.columns]
        
        st.dataframe(display_df[display_cols], use_container_width=True)

# Equity Curve
st.header("Equity Curve")

# Load equity curve (simulated for now)
try:
    equity_file = os.path.join(DATA_DIR, "performance", "equity_curve.json")
    
    if os.path.exists(equity_file):
        with open(equity_file, 'r') as f:
            equity_data = json.load(f)
    else:
        # Generate simulated equity curve
        now = datetime.datetime.now()
        dates = [(now - datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
        equity = [10000 + i * 100 + np.random.normal(0, 500) for i in range(30)]
        equity_data = [{"timestamp": date, "equity": value} for date, value in zip(dates, equity)]
except Exception as e:
    st.error(f"Error loading equity curve data: {str(e)}")
    
    # Generate simulated equity curve
    now = datetime.datetime.now()
    dates = [(now - datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
    equity = [10000 + i * 100 + np.random.normal(0, 500) for i in range(30)]
    equity_data = [{"timestamp": date, "equity": value} for date, value in zip(dates, equity)]

# Convert to DataFrame
if equity_data:
    try:
        equity_df = pd.DataFrame(equity_data)
        
        # Convert timestamp to datetime
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(equity_df['timestamp'], equity_df['equity'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.set_title('Portfolio Equity Curve')
        ax.grid(True)
        
        # Format x-axis
        fig.autofmt_xdate()
        
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting equity curve: {str(e)}")
else:
    st.info("No equity curve data available")

# Strategy Performance
st.header("Strategy Performance")

# Load strategy performance (simulated for now)
try:
    strategy_file = os.path.join(DATA_DIR, "performance", "strategy_performance.json")
    
    if os.path.exists(strategy_file):
        with open(strategy_file, 'r') as f:
            strategy_data = json.load(f)
    else:
        # Simulated strategy data
        strategy_data = {
            "moving_average_crossover": {
                "trades": 15,
                "winning_trades": 9,
                "losing_trades": 6,
                "pnl": 1200,
                "win_rate": 0.6,
                "avg_win": 200,
                "avg_loss": -100,
                "profit_factor": 2.0
            },
            "rsi": {
                "trades": 12,
                "winning_trades": 7,
                "losing_trades": 5,
                "pnl": 800,
                "win_rate": 0.583,
                "avg_win": 180,
                "avg_loss": -110,
                "profit_factor": 1.8
            },
            "trend_following": {
                "trades": 8,
                "winning_trades": 3,
                "losing_trades": 5,
                "pnl": 200,
                "win_rate": 0.375,
                "avg_win": 300,
                "avg_loss": -120,
                "profit_factor": 1.5
            }
        }
except Exception as e:
    st.error(f"Error loading strategy performance data: {str(e)}")
    
    # Simulated strategy data
    strategy_data = {
        "moving_average_crossover": {
            "trades": 15,
            "winning_trades": 9,
            "losing_trades": 6,
            "pnl": 1200,
            "win_rate": 0.6,
            "avg_win": 200,
            "avg_loss": -100,
            "profit_factor": 2.0
        },
        "rsi": {
            "trades": 12,
            "winning_trades": 7,
            "losing_trades": 5,
            "pnl": 800,
            "win_rate": 0.583,
            "avg_win": 180,
            "avg_loss": -110,
            "profit_factor": 1.8
        },
        "trend_following": {
            "trades": 8,
            "winning_trades": 3,
            "losing_trades": 5,
            "pnl": 200,
            "win_rate": 0.375,
            "avg_win": 300,
            "avg_loss": -120,
            "profit_factor": 1.5
        }
    }

# Convert to DataFrame
if strategy_data:
    try:
        strategy_list = []
        for strategy, metrics in strategy_data.items():
            metrics['strategy'] = strategy
            strategy_list.append(metrics)
        
        strategy_df = pd.DataFrame(strategy_list)
        
        # Display as table
        if not strategy_df.empty:
            # Format columns
            for col in ['win_rate']:
                if col in strategy_df.columns:
                    strategy_df[col] = strategy_df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "")
                    
            for col in ['pnl', 'avg_win', 'avg_loss']:
                if col in strategy_df.columns:
                    strategy_df[col] = strategy_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "")
            
            # Select columns to display
            display_cols = ['strategy', 'trades', 'winning_trades', 'losing_trades', 
                          'pnl', 'win_rate', 'profit_factor']
            display_cols = [col for col in display_cols if col in strategy_df.columns]
            
            st.dataframe(strategy_df[display_cols], use_container_width=True)
            
            # Plot strategy comparison
            if 'trades' in strategy_df.columns and len(strategy_df) > 1:
                # Plot
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                # Convert win_rate back to float
                if 'win_rate' in strategy_df.columns:
                    strategy_df['win_rate_float'] = strategy_df['win_rate'].apply(
                        lambda x: float(x.strip('%')) / 100 if isinstance(x, str) else x
                    )
                
                # Convert pnl back to float
                if 'pnl' in strategy_df.columns:
                    strategy_df['pnl_float'] = strategy_df['pnl'].apply(
                        lambda x: float(x.strip('$')) if isinstance(x, str) else x
                    )
                
                # Bar width
                width = 0.3
                
                # X positions
                x = np.arange(len(strategy_df))
                
                # Plot win rate
                if 'win_rate_float' in strategy_df.columns:
                    ax1.bar(x - width/2, strategy_df['win_rate_float'] * 100, width, label='Win Rate (%)', color='blue', alpha=0.7)
                    ax1.set_ylabel('Win Rate (%)')
                    ax1.set_ylim(0, 100)
                
                # Create second y-axis
                ax2 = ax1.twinx()
                
                # Plot profit factor
                if 'profit_factor' in strategy_df.columns:
                    ax2.bar(x + width/2, strategy_df['profit_factor'], width, label='Profit Factor', color='green', alpha=0.7)
                    ax2.set_ylabel('Profit Factor')
                    ax2.set_ylim(0, max(5, strategy_df['profit_factor'].max() * 1.2))
                
                # Add labels
                ax1.set_xticks(x)
                ax1.set_xticklabels(strategy_df['strategy'], rotation=45, ha='right')
                ax1.set_title('Strategy Comparison')
                
                # Add P&L annotations
                if 'pnl_float' in strategy_df.columns:
                    for i, pnl in enumerate(strategy_df['pnl_float']):
                        ax1.annotate(f"${pnl:.0f}", 
                                   xy=(i, 5),
                                   ha='center',
                                   va='bottom')
                
                # Add legends
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                
                plt.tight_layout()
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Error displaying strategy performance: {str(e)}")
else:
    st.info("No strategy performance data available")

# Button actions
if start_button:
    st.info("Running trading cycle... (this is a demo, not actually executing trades)")
    
    # Show a progress bar
    progress_bar = st.progress(0)
    for i in range(101):
        time.sleep(0.03)
        progress_bar.progress(i)
    
    st.success("Trading cycle completed!")
    
    # Update simulated portfolio data
    st.session_state.portfolio_value = 112500.00
    st.session_state.buying_power = 62500.00
    st.session_state.positions = {
        "AAPL": {
            "market": "US",
            "symbol": "AAPL",
            "quantity": 50,
            "entry_price": 175.25,
            "current_price": 178.50,
            "market_value": 8925.00,
            "unrealized_pl": 162.50,
            "unrealized_plpc": 0.0185
        },
        "MSFT": {
            "market": "US",
            "symbol": "MSFT",
            "quantity": 30,
            "entry_price": 325.75,
            "current_price": 330.25,
            "market_value": 9907.50,
            "unrealized_pl": 135.00,
            "unrealized_plpc": 0.0138
        },
        "RELIANCE-EQ": {
            "market": "India",
            "symbol": "RELIANCE-EQ",
            "quantity": 100,
            "entry_price": 2450.00,
            "current_price": 2480.50,
            "market_value": 248050.00,
            "unrealized_pl": 3050.00,
            "unrealized_plpc": 0.0124
        }
    }
    
    # Rerun to update display
    st.experimental_rerun()

if schedule_button:
    st.info("Bot scheduled for trading at market hours. (Demo only)")

if stop_button:
    st.warning("Bot stopped. (Demo only)")

# Disclaimer
st.markdown("---")
st.caption("""
**Disclaimer**: This is a demo application. In a real deployment, you would need to set environment variables or configure API keys securely. 
The trading bot is for educational purposes only and should not be used with real funds without proper risk management and testing.
""")

# Auto-refresh every 5 minutes
st.markdown(
    """
    <script>
        setTimeout(function() {
            window.location.reload();
        }, 300000);
    </script>
    """,
    unsafe_allow_html=True
)