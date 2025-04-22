"""
Indian Market Connector

This module provides utility functions to integrate Indian market strategies with the rest of the application.
It connects Angel One API with the strategy engine and handles Indian market specific operations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import traceback

from .angel_one_api import AngelOneAPI, initialize_angel_one_api, check_angel_one_credentials
from .indian_strategies import (
    get_indian_market_strategies, list_indian_strategies, 
    get_strategy_details, run_indian_strategy,
    nifty_bank_nifty_momentum, nifty_gap_trading, vwap_nse_intraday,
    option_writing_nifty, fii_dii_flow_strategy, nifty_bank_nifty_correlation
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('indian_market_connector')

def get_indian_market_symbols():
    """
    Get a list of available Indian market symbols
    
    Returns:
        list: List of symbol strings
    """
    # Common Indian market symbols
    common_symbols = [
        "NIFTY 50", "BANKNIFTY", "RELIANCE-EQ", "TCS-EQ", "HDFCBANK-EQ", "INFY-EQ", 
        "ICICIBANK-EQ", "KOTAKBANK-EQ", "HINDUNILVR-EQ", "SBIN-EQ", "BAJFINANCE-EQ",
        "BHARTIARTL-EQ", "ITC-EQ", "ASIANPAINT-EQ", "MARUTI-EQ", "TITAN-EQ"
    ]
    
    # TODO: Expand this list with actual symbols from Angel One API
    # Try to get more symbols from API if available
    try:
        api = initialize_angel_one_api()
        if api and api.authenticated:
            # This is a placeholder for actual API integration to get more symbols
            # In a real implementation, you'd fetch these from the Angel One API
            pass
    except Exception as e:
        logger.error(f"Error fetching Indian market symbols: {e}")
        logger.error(traceback.format_exc())
    
    return common_symbols

def get_indian_historical_data(symbol, exchange="NSE", interval="ONE_DAY", days=30):
    """
    Get historical data for an Indian market symbol
    
    Args:
        symbol (str): Trading symbol (e.g. "RELIANCE-EQ")
        exchange (str): Exchange (NSE, BSE)
        interval (str): Candle interval (ONE_MINUTE, FIVE_MINUTE, etc.)
        days (int): Number of days of historical data
        
    Returns:
        DataFrame: OHLCV data for the symbol
    """
    try:
        # Check if Angel One credentials are available
        has_credentials, _ = check_angel_one_credentials()
        if not has_credentials:
            st.warning("Angel One API credentials required for Indian market data. Please add them in Settings.")
            return pd.DataFrame()
        
        # Initialize API
        api = initialize_angel_one_api()
        
        # Set date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # Format dates
        from_date_str = from_date.strftime("%Y-%m-%d")
        to_date_str = to_date.strftime("%Y-%m-%d")
        
        # Get data from Angel One
        df = api.get_historical_data(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            from_date=from_date_str,
            to_date=to_date_str
        )
        
        return df
    except Exception as e:
        logger.error(f"Error getting Indian historical data: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Failed to fetch Indian market data: {str(e)}")
        return pd.DataFrame()

def run_indian_backtest(strategy_name, symbol, exchange="NSE", interval="ONE_DAY", days=180, params=None):
    """
    Run backtest for an Indian market strategy
    
    Args:
        strategy_name (str): Name of the Indian strategy to backtest
        symbol (str): Trading symbol
        exchange (str): Exchange (NSE, BSE)
        interval (str): Candle interval
        days (int): Number of days for backtest
        params (dict): Strategy parameters
        
    Returns:
        tuple: (backtest_results_df, performance_metrics)
    """
    try:
        # Get historical data
        data = get_indian_historical_data(symbol, exchange, interval, days)
        if data.empty:
            st.warning("No historical data available for backtest.")
            return pd.DataFrame(), {}
        
        # Get strategy details
        strategy_details = get_strategy_details(strategy_name)
        if not strategy_details:
            st.warning(f"Strategy '{strategy_name}' not found.")
            return pd.DataFrame(), {}
        
        # Run strategy
        params = params or {}
        results = run_indian_strategy(strategy_name, data, **params)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(results)
        
        return results, metrics
    except Exception as e:
        logger.error(f"Error running Indian backtest: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Backtest failed: {str(e)}")
        return pd.DataFrame(), {}

def calculate_performance_metrics(results_df):
    """
    Calculate performance metrics from backtest results
    
    Args:
        results_df (DataFrame): Backtest results with signals
        
    Returns:
        dict: Performance metrics
    """
    # Check if we have data and signals
    if results_df.empty or 'signal' not in results_df.columns:
        return {
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_return': 0,
            'annualized_return': 0,
            'num_trades': 0
        }
    
    # Get price data and signals
    df = results_df.copy()
    
    # Generate positions (1 for long, -1 for short, 0 for flat)
    df['position'] = 0
    
    # Find where signals occur
    signal_days = df[df['signal'] != 0].index
    
    if len(signal_days) == 0:
        return {
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_return': 0,
            'annualized_return': 0,
            'num_trades': 0
        }
    
    # For each signal, hold the position until the next opposing signal
    current_position = 0
    for i, day in enumerate(signal_days):
        signal = df.loc[day, 'signal']
        
        # If signal matches current position, continue
        if (signal > 0 and current_position > 0) or (signal < 0 and current_position < 0):
            continue
            
        # Close current position and open new one
        if signal != 0:
            current_position = signal
            
        # Set position from this day until next signal
        next_signal_idx = i + 1 if i + 1 < len(signal_days) else len(df)
        if next_signal_idx < len(signal_days):
            end_idx = df.index.get_loc(signal_days[next_signal_idx])
        else:
            end_idx = len(df)
        
        # Set position from current signal until next signal
        start_idx = df.index.get_loc(day)
        df.iloc[start_idx:end_idx, df.columns.get_loc('position')] = current_position
    
    # Calculate daily returns based on position
    df['returns'] = df['close'].pct_change() * df['position'].shift(1)
    df.iloc[0, df.columns.get_loc('returns')] = 0  # Set first day's return to 0
    
    # Calculate cumulative returns
    df['cum_returns'] = (1 + df['returns']).cumprod()
    
    # Calculate drawdowns
    df['peak'] = df['cum_returns'].cummax()
    df['drawdown'] = (df['cum_returns'] - df['peak']) / df['peak']
    
    # Find trades (position changes)
    df['trade'] = df['position'].diff().abs()
    df.iloc[0, df.columns.get_loc('trade')] = 1 if df.iloc[0, df.columns.get_loc('position')] != 0 else 0
    
    # Calculate trade returns
    df['trade_end'] = df['position'].diff().abs() > 0
    trade_ends = df[df['trade_end']].index
    
    wins = 0
    losses = 0
    profits = 0
    losses_sum = 0
    
    for i in range(1, len(trade_ends)):
        # Get trade period
        start_date = trade_ends[i-1]
        end_date = trade_ends[i]
        
        # Calculate trade return
        trade_return = df.loc[end_date, 'cum_returns'] / df.loc[start_date, 'cum_returns'] - 1
        
        if trade_return > 0:
            wins += 1
            profits += trade_return
        elif trade_return < 0:
            losses += 1
            losses_sum += abs(trade_return)
    
    # Calculate metrics
    num_trades = wins + losses
    win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
    profit_factor = profits / losses_sum if losses_sum > 0 else float('inf')
    max_drawdown = abs(df['drawdown'].min() * 100)
    total_return = (df['cum_returns'].iloc[-1] - 1) * 100
    
    # Calculate Sharpe ratio
    risk_free_rate = 0.03  # Assumed 3% annual risk-free rate
    daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = df['returns'] - daily_risk_free
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
    
    # Calculate annualized return
    days = (df.index[-1] - df.index[0]).days
    years = days / 365
    annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
    
    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'num_trades': num_trades
    }

def get_active_indian_strategies():
    """
    Get a list of active Indian market strategies
    
    Returns:
        list: List of strategy dictionaries
    """
    if 'indian_strategies' not in st.session_state:
        st.session_state.indian_strategies = []
    
    return st.session_state.indian_strategies

def add_indian_strategy(name, description, symbol, strategy_type, parameters=None, status="Active"):
    """
    Add a new Indian market strategy to the active strategies list
    
    Args:
        name (str): Strategy name
        description (str): Strategy description
        symbol (str): Trading symbol
        strategy_type (str): Type of strategy
        parameters (dict): Strategy parameters
        status (str): Strategy status (Active/Paused)
        
    Returns:
        dict: The newly added strategy
    """
    if 'indian_strategies' not in st.session_state:
        st.session_state.indian_strategies = []
    
    # Create ID for new strategy
    strategy_id = len(st.session_state.indian_strategies) + 1
    
    # Create strategy dictionary
    strategy = {
        "id": strategy_id,
        "name": name,
        "description": description,
        "symbols": [symbol] if isinstance(symbol, str) else symbol,
        "type": strategy_type,
        "status": status,
        "parameters": parameters or {},
        "risk": {
            "position_size": 10,
            "stop_loss": 5,
            "take_profit": 15
        },
        "performance": {
            "win_rate": 0,
            "profit_factor": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0
        }
    }
    
    # Add to session state
    st.session_state.indian_strategies.append(strategy)
    
    return strategy

def update_indian_strategy(strategy_id, **kwargs):
    """
    Update an existing Indian market strategy
    
    Args:
        strategy_id (int): ID of the strategy to update
        **kwargs: Fields to update
        
    Returns:
        bool: True if successful, False otherwise
    """
    if 'indian_strategies' not in st.session_state:
        return False
    
    # Find strategy by ID
    for i, strategy in enumerate(st.session_state.indian_strategies):
        if strategy["id"] == strategy_id:
            # Update fields
            for key, value in kwargs.items():
                strategy[key] = value
            
            # Update in session state
            st.session_state.indian_strategies[i] = strategy
            return True
    
    return False

def delete_indian_strategy(strategy_id):
    """
    Delete an Indian market strategy
    
    Args:
        strategy_id (int): ID of the strategy to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    if 'indian_strategies' not in st.session_state:
        return False
    
    # Find strategy by ID
    for i, strategy in enumerate(st.session_state.indian_strategies):
        if strategy["id"] == strategy_id:
            # Remove from session state
            st.session_state.indian_strategies.pop(i)
            return True
    
    return False

def get_indian_market_data_sources():
    """
    Get a list of available Indian market data sources
    
    Returns:
        list: List of data source dictionaries
    """
    return [
        {"id": "angel_one", "name": "Angel One", "description": "Comprehensive Indian broker API"},
        {"id": "nse", "name": "National Stock Exchange (NSE)", "description": "Official NSE data"},
        {"id": "bse", "name": "Bombay Stock Exchange (BSE)", "description": "Official BSE data"},
        {"id": "five_paisa", "name": "5paisa", "description": "Alternative Indian broker API"}
    ]