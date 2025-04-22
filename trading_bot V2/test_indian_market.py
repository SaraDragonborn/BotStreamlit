"""
Test script for Indian market functionality
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
from datetime import datetime, timedelta

# Add the parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from streamlit_app.utils.angel_one_api import AngelOneAPI, test_angel_one_connection
from streamlit_app.utils.fixed_api import (
    get_indian_symbols, get_indian_data, 
    backtest_indian_strategy, get_indian_strategies
)

# Function to display stock data
def display_data(df, title):
    if df.empty:
        st.warning("No data available for display")
        return

    st.subheader(title)
    st.dataframe(df.head())

    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Candlesticks"
    )])

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

# Main application
def main():
    st.set_page_config(page_title="Indian Market Test", page_icon="🇮🇳", layout="wide")
    
    st.title("Indian Market Trading Functionality Test")
    st.markdown("""
    This is a test script to verify that the Angel One API integration is working correctly for Indian market trading.
    It checks connection, retrieves market data, and tests strategy backtesting for Indian market symbols.
    """)
    
    # API connection check
    st.header("1. Angel One API Connection")
    
    # Load credentials from environment variables or session state
    api_key = os.environ.get('ANGEL_ONE_API_KEY') or st.session_state.get('angel_one_api_key', '')
    client_id = os.environ.get('ANGEL_ONE_CLIENT_ID') or st.session_state.get('angel_one_client_id', '')
    password = os.environ.get('ANGEL_ONE_PASSWORD') or st.session_state.get('angel_one_password', '')
    historical_api_key = os.environ.get('ANGEL_ONE_HISTORICAL_API_KEY') or st.session_state.get('angel_one_historical_api_key', '')
    market_feed_api_key = os.environ.get('ANGEL_ONE_MARKET_FEED_API_KEY') or st.session_state.get('angel_one_market_feed_api_key', '')
    
    # Display masked credentials
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"API Key: {'✓ Present' if api_key else '✗ Missing'}")
        st.info(f"Client ID: {'✓ Present' if client_id else '✗ Missing'}")
        st.info(f"Password: {'✓ Present' if password else '✗ Missing'}")
    
    with col2:
        st.info(f"Historical API Key: {'✓ Present' if historical_api_key else '✗ Missing'}")
        st.info(f"Market Feed API Key: {'✓ Present' if market_feed_api_key else '✗ Missing'}")
    
    if st.button("Test Angel One Connection"):
        with st.spinner("Testing connection to Angel One API..."):
            success, message = test_angel_one_connection()
            
            if success:
                st.success(f"Connection successful: {message}")
            else:
                st.error(f"Connection failed: {message}")
                st.markdown("""
                Please make sure you have added your Angel One API credentials in the Settings page.
                Go to: Settings > API Connections > Angel One API Settings
                """)
    
    # Available symbols
    st.header("2. Available Indian Market Symbols")
    
    if st.button("Get Available Symbols"):
        with st.spinner("Retrieving symbols..."):
            try:
                symbols = get_indian_symbols()
                st.write(f"Found {len(symbols)} Indian market symbols:")
                st.write(symbols)
            except Exception as e:
                st.error(f"Error retrieving symbols: {str(e)}")
    
    # Market Data Retrieval
    st.header("3. Historical Data Retrieval")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.selectbox(
            "Select Symbol",
            options=["NIFTY 50", "BANKNIFTY", "RELIANCE-EQ", "TCS-EQ", "HDFCBANK-EQ", "INFY-EQ"]
        )
    
    with col2:
        exchange = st.selectbox(
            "Exchange",
            options=["NSE", "BSE"],
            index=0
        )
    
    with col3:
        interval = st.selectbox(
            "Interval",
            options=["ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "THIRTY_MINUTE", "ONE_HOUR", "ONE_DAY"],
            index=5  # Default to ONE_DAY
        )
    
    days = st.slider("Days of History", min_value=5, max_value=180, value=30)
    
    if st.button("Get Historical Data"):
        with st.spinner(f"Retrieving data for {symbol} on {exchange} with interval {interval}..."):
            try:
                df = get_indian_data(symbol, exchange, interval, days)
                
                if df is not None and not df.empty:
                    display_data(df, f"{symbol} Historical Data ({interval})")
                else:
                    st.warning(f"No data returned for {symbol}")
            except Exception as e:
                st.error(f"Error retrieving historical data: {str(e)}")
    
    # Strategy Backtesting
    st.header("4. Strategy Backtesting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        strategy_name = st.selectbox(
            "Select Strategy",
            options=["nifty_bank_nifty_momentum", "nifty_gap_trading", "vwap_nse_intraday", "option_writing_nifty"]
        )
    
    with col2:
        backtest_symbol = st.selectbox(
            "Symbol for Backtest",
            options=["NIFTY 50", "BANKNIFTY", "RELIANCE-EQ", "TCS-EQ", "HDFCBANK-EQ", "INFY-EQ"],
            key="backtest_symbol"
        )
    
    backtest_days = st.slider("Backtest Period (Days)", min_value=30, max_value=365, value=90)
    
    # Strategy parameters
    st.subheader("Strategy Parameters")
    
    if strategy_name == "nifty_bank_nifty_momentum":
        col1, col2 = st.columns(2)
        with col1:
            fast_period = st.number_input("Fast Period", min_value=5, max_value=20, value=9)
        with col2:
            slow_period = st.number_input("Slow Period", min_value=15, max_value=50, value=21)
        
        params = {
            "fast_period": fast_period,
            "slow_period": slow_period
        }
    
    elif strategy_name == "vwap_nse_intraday":
        deviation = st.slider("VWAP Deviation", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
        params = {"deviation": deviation}
    
    elif strategy_name == "option_writing_nifty":
        col1, col2 = st.columns(2)
        with col1:
            days_to_expiry = st.number_input("Days to Expiry", min_value=1, max_value=30, value=5)
        with col2:
            iv_threshold = st.number_input("IV Threshold", min_value=15, max_value=40, value=20)
        
        params = {
            "days_to_expiry": days_to_expiry,
            "iv_threshold": iv_threshold
        }
    
    else:  # nifty_gap_trading
        params = {}  # No parameters for gap trading
    
    if st.button("Run Backtest"):
        with st.spinner(f"Running {strategy_name} backtest on {backtest_symbol}..."):
            try:
                results_df, metrics = backtest_indian_strategy(
                    strategy_name=strategy_name,
                    symbol=backtest_symbol,
                    exchange="NSE",
                    interval="ONE_DAY",
                    days=backtest_days,
                    params=params
                )
                
                if results_df is not None and not results_df.empty:
                    st.subheader("Backtest Results")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
                    col2.metric("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
                    col3.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    col4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
                    
                    # Display equity curve if available
                    if 'cum_returns' in results_df.columns:
                        initial_capital = 100000  # 1 lakh rupees
                        equity = [initial_capital * (1 + r) for r in results_df['cum_returns']]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=results_df.index,
                            y=equity,
                            mode='lines',
                            name='Portfolio Value',
                            line=dict(color='#1E88E5', width=2)
                        ))
                        
                        fig.update_layout(
                            title=f"Equity Curve: {strategy_name} on {backtest_symbol}",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value (₹)",
                            height=400,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display signals
                    if 'signal' in results_df.columns:
                        signals_df = results_df[['open', 'high', 'low', 'close', 'signal']].copy()
                        st.subheader("Trading Signals")
                        st.dataframe(signals_df.tail(20))
                        
                        # Plot price with buy/sell signals
                        fig = go.Figure()
                        
                        # Add price line
                        fig.add_trace(go.Scatter(
                            x=signals_df.index,
                            y=signals_df['close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='#757575', width=1)
                        ))
                        
                        # Add buy signals
                        buy_signals = signals_df[signals_df['signal'] > 0]
                        if not buy_signals.empty:
                            fig.add_trace(go.Scatter(
                                x=buy_signals.index,
                                y=buy_signals['close'],
                                mode='markers',
                                name='Buy Signal',
                                marker=dict(color='green', size=10, symbol='triangle-up')
                            ))
                        
                        # Add sell signals
                        sell_signals = signals_df[signals_df['signal'] < 0]
                        if not sell_signals.empty:
                            fig.add_trace(go.Scatter(
                                x=sell_signals.index,
                                y=sell_signals['close'],
                                mode='markers',
                                name='Sell Signal',
                                marker=dict(color='red', size=10, symbol='triangle-down')
                            ))
                        
                        fig.update_layout(
                            title=f"Trading Signals: {strategy_name} on {backtest_symbol}",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            height=400,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No backtest results returned for {strategy_name} on {backtest_symbol}")
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")

if __name__ == "__main__":
    main()