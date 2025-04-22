"""
Indian Market-Specific Trading Strategies

This module contains trading strategies specifically designed for the 
Indian stock market, including NSE and BSE focused approaches. These
strategies are optimized for Indian market conditions and regulations.
"""

import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import streamlit as st

def nifty_bank_nifty_momentum(data, fast_period=9, slow_period=21):
    """
    Momentum-based strategy specifically for Nifty and Bank Nifty indices
    Uses EMA crossovers and RSI for signal generation.
    
    Args:
        data (DataFrame): OHLCV dataframe with price data
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        
    Returns:
        DataFrame: Original data with added signal columns
    """
    # Make a copy of data to avoid modifying the original
    df = data.copy()
    
    # Calculate EMAs
    df['fast_ema'] = talib.EMA(df['close'], timeperiod=fast_period)
    df['slow_ema'] = talib.EMA(df['close'], timeperiod=slow_period)
    
    # Calculate RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    
    # Volume Surge detection
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Generate Signals
    df['signal'] = 0
    
    # Buy Signal: Fast EMA crosses above Slow EMA, RSI > 50, and increased volume
    buy_condition = (
        (df['fast_ema'] > df['slow_ema']) & 
        (df['fast_ema'].shift(1) <= df['slow_ema'].shift(1)) & 
        (df['rsi'] > 50) &
        (df['volume_ratio'] > 1.5)
    )
    df.loc[buy_condition, 'signal'] = 1
    
    # Sell Signal: Fast EMA crosses below Slow EMA or RSI > 70
    sell_condition = (
        ((df['fast_ema'] < df['slow_ema']) & (df['fast_ema'].shift(1) >= df['slow_ema'].shift(1))) | 
        (df['rsi'] > 70)
    )
    df.loc[sell_condition, 'signal'] = -1
    
    return df

def nifty_gap_trading(data):
    """
    Gap Trading strategy for Nifty - specifically designed for Indian market opening gaps
    
    Args:
        data (DataFrame): OHLCV dataframe with price data
        
    Returns:
        DataFrame: Original data with added signal columns
    """
    df = data.copy()
    
    # Calculate gaps
    df['gap'] = df['open'] - df['close'].shift(1)
    df['gap_percent'] = (df['gap'] / df['close'].shift(1)) * 100
    
    # Generate signals
    df['signal'] = 0
    
    # Buy Signal: Negative gap (gap down) exceeding 0.5%
    buy_condition = df['gap_percent'] < -0.5
    df.loc[buy_condition, 'signal'] = 1
    
    # Sell Signal: Positive gap (gap up) exceeding 0.5%
    sell_condition = df['gap_percent'] > 0.5
    df.loc[sell_condition, 'signal'] = -1
    
    return df

def vwap_nse_intraday(data, deviation=1.5):
    """
    VWAP-based intraday strategy specifically for NSE stocks
    Uses Volume Weighted Average Price with deviation bands
    
    Args:
        data (DataFrame): OHLCV dataframe with price data
        deviation (float): Standard deviation multiplier for bands
        
    Returns:
        DataFrame: Original data with added signal columns
    """
    df = data.copy()
    
    # Calculate VWAP
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # Calculate standard deviation
    df['sd'] = (((df['high'] + df['low'] + df['close']) / 3 - df['vwap']) ** 2 * df['volume']).cumsum()
    df['sd'] = np.sqrt(df['sd'] / df['volume'].cumsum())
    
    # Upper and lower bands
    df['upper_band'] = df['vwap'] + deviation * df['sd']
    df['lower_band'] = df['vwap'] - deviation * df['sd']
    
    # Generate signals
    df['signal'] = 0
    
    # Buy Signal: Price crosses below lower band
    buy_condition = (df['close'] < df['lower_band']) & (df['close'].shift(1) >= df['lower_band'].shift(1))
    df.loc[buy_condition, 'signal'] = 1
    
    # Sell Signal: Price crosses above upper band
    sell_condition = (df['close'] > df['upper_band']) & (df['close'].shift(1) <= df['upper_band'].shift(1))
    df.loc[sell_condition, 'signal'] = -1
    
    return df

def option_writing_nifty(data, days_to_expiry=5, iv_threshold=20):
    """
    Option Writing strategy for Nifty - specifically designed for NSE F&O
    Used for weekly or monthly expiry options strategies with IV considerations
    
    Args:
        data (DataFrame): OHLCV dataframe with price data
        days_to_expiry (int): Days remaining to expiry
        iv_threshold (float): Implied volatility threshold
        
    Returns:
        DataFrame: Original data with added signal columns
    """
    df = data.copy()
    
    # This is a placeholder for IV calculation - in a real implementation,
    # you would need actual option chain data with IV values
    # Here we're using a simple approximation based on historical volatility
    df['20d_volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
    df['iv_proxy'] = df['20d_volatility'] * 1.1  # IV is typically higher than historical vol
    
    # Calculate signals
    df['signal'] = 0
    
    # For Option Writing, we sell options (write) when:
    # 1. IV is high (above threshold)
    # 2. We're close to expiry (premium decay accelerates)
    # 3. Market is range-bound (no strong trend)
    
    # Calculate if market is range-bound using Bollinger Bands Width
    df['sma20'] = talib.SMA(df['close'], timeperiod=20)
    df['stddev'] = df['close'].rolling(window=20).std()
    df['bb_width'] = (df['sma20'] + 2 * df['stddev'] - (df['sma20'] - 2 * df['stddev'])) / df['sma20']
    
    # Write signals (sell options) when IV is high and market is not trending strongly
    write_condition = (df['iv_proxy'] > iv_threshold) & (df['bb_width'] < df['bb_width'].rolling(window=20).mean())
    df.loc[write_condition, 'signal'] = -1  # -1 indicates writing/selling options
    
    # Exit signals (buy back options) when IV contracts or market starts trending
    exit_condition = (df['iv_proxy'] < iv_threshold * 0.8) | (df['bb_width'] > df['bb_width'].rolling(window=20).mean() * 1.2)
    df.loc[exit_condition, 'signal'] = 1  # 1 indicates buying back options
    
    return df

def fii_dii_flow_strategy(data, fii_data=None, dii_data=None):
    """
    Strategy based on FII (Foreign Institutional Investors) and 
    DII (Domestic Institutional Investors) fund flows - unique to Indian markets
    
    Args:
        data (DataFrame): OHLCV dataframe with price data
        fii_data (DataFrame): FII buy/sell data
        dii_data (DataFrame): DII buy/sell data
        
    Returns:
        DataFrame: Original data with added signal columns
    """
    df = data.copy()
    
    # Generate placeholder FII/DII data if not provided
    # In a real implementation, you would use actual FII/DII data from a proper source
    if fii_data is None:
        # Create synthetic FII data based on price movements
        # This is just a placeholder - real implementation would use actual data
        df['fii_net'] = df['close'].diff().rolling(window=5).sum() * 10000000  # Scale factor
    else:
        # Merge actual FII data
        df['fii_net'] = fii_data['net_value']
    
    if dii_data is None:
        # Create synthetic DII data - often inversely correlated with FII
        df['dii_net'] = -df['close'].diff().rolling(window=5).sum() * 8000000  # Scale factor
    else:
        # Merge actual DII data
        df['dii_net'] = dii_data['net_value']
    
    # Calculate moving averages of FII and DII flows
    df['fii_net_ma5'] = df['fii_net'].rolling(window=5).mean()
    df['dii_net_ma5'] = df['dii_net'].rolling(window=5).mean()
    
    # Generate signals
    df['signal'] = 0
    
    # Buy Signal: Strong FII buying (more significant than DII)
    buy_condition = (df['fii_net'] > 0) & (df['fii_net'] > df['fii_net_ma5'] * 1.5) & (df['fii_net'].abs() > df['dii_net'].abs())
    df.loc[buy_condition, 'signal'] = 1
    
    # Sell Signal: Strong FII selling (more significant than DII)
    sell_condition = (df['fii_net'] < 0) & (df['fii_net'] < df['fii_net_ma5'] * 1.5) & (df['fii_net'].abs() > df['dii_net'].abs())
    df.loc[sell_condition, 'signal'] = -1
    
    return df

def nifty_bank_nifty_correlation(nifty_data, bank_nifty_data, correlation_window=20, threshold=0.7):
    """
    Strategy based on correlation divergence between Nifty and Bank Nifty
    
    Args:
        nifty_data (DataFrame): OHLCV dataframe for Nifty
        bank_nifty_data (DataFrame): OHLCV dataframe for Bank Nifty
        correlation_window (int): Window for correlation calculation
        threshold (float): Correlation threshold for signals
        
    Returns:
        tuple: (Nifty data with signals, Bank Nifty data with signals)
    """
    nifty_df = nifty_data.copy()
    banknifty_df = bank_nifty_data.copy()
    
    # Calculate returns
    nifty_df['returns'] = nifty_df['close'].pct_change()
    banknifty_df['returns'] = banknifty_df['close'].pct_change()
    
    # Calculate rolling correlation
    merged_df = pd.DataFrame({
        'nifty': nifty_df['returns'],
        'banknifty': banknifty_df['returns']
    })
    merged_df['correlation'] = merged_df['nifty'].rolling(correlation_window).corr(merged_df['banknifty'])
    
    # Add correlation back to both dataframes
    nifty_df['correlation'] = merged_df['correlation']
    banknifty_df['correlation'] = merged_df['correlation']
    
    # Calculate signal for Nifty
    nifty_df['signal'] = 0
    # Buy Nifty when correlation drops (divergence) and Nifty is underperforming
    nifty_buy = (nifty_df['correlation'] < threshold) & (nifty_df['returns'] < banknifty_df['returns'])
    nifty_df.loc[nifty_buy, 'signal'] = 1
    # Sell Nifty when correlation is high and Nifty is outperforming
    nifty_sell = (nifty_df['correlation'] > 0.9) & (nifty_df['returns'] > banknifty_df['returns'])
    nifty_df.loc[nifty_sell, 'signal'] = -1
    
    # Calculate signal for Bank Nifty
    banknifty_df['signal'] = 0
    # Buy Bank Nifty when correlation drops and Bank Nifty is underperforming
    banknifty_buy = (banknifty_df['correlation'] < threshold) & (banknifty_df['returns'] < nifty_df['returns'])
    banknifty_df.loc[banknifty_buy, 'signal'] = 1
    # Sell Bank Nifty when correlation is high and Bank Nifty is outperforming
    banknifty_sell = (banknifty_df['correlation'] > 0.9) & (banknifty_df['returns'] > nifty_df['returns'])
    banknifty_df.loc[banknifty_sell, 'signal'] = -1
    
    return nifty_df, banknifty_df

def indian_market_hours_filter(signals_df, trading_calendar=None):
    """
    Filter signals based on Indian market trading hours
    NSE/BSE market hours: 9:15 AM - 3:30 PM IST, Monday-Friday
    
    Args:
        signals_df (DataFrame): DataFrame with signal column
        trading_calendar (DataFrame, optional): Trading calendar with holiday information
        
    Returns:
        DataFrame: Filtered signals dataframe
    """
    df = signals_df.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            raise ValueError("DataFrame must have a datetime index or a 'timestamp' column")
    
    # Convert to IST timezone if not already
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    elif df.index.tz.zone != 'Asia/Kolkata':
        df.index = df.index.tz_convert('Asia/Kolkata')
    
    # Filter for IST trading hours (9:15 AM - 3:30 PM)
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_valid'] = ((df['hour'] > 9) | ((df['hour'] == 9) & (df['minute'] >= 15))) & (df['hour'] < 15 | ((df['hour'] == 15) & (df['minute'] <= 30)))
    
    # Filter for weekdays (Monday-Friday)
    df['weekday'] = df.index.weekday
    df['weekday_valid'] = df['weekday'] < 5  # 0-4 are Monday to Friday
    
    # Filter for holidays if calendar provided
    if trading_calendar is not None:
        # Convert trading_calendar dates to datetime if not already
        trading_calendar['date'] = pd.to_datetime(trading_calendar['date'])
        
        # Create date column in df
        df['date'] = df.index.date
        
        # Filter out holidays
        holidays = set(trading_calendar[trading_calendar['is_holiday']]['date'].dt.date)
        df['not_holiday'] = ~df['date'].isin(holidays)
    else:
        df['not_holiday'] = True
    
    # Apply all filters
    df['valid_trading_time'] = df['time_valid'] & df['weekday_valid'] & df['not_holiday']
    
    # Filter signals to only valid trading times
    df.loc[~df['valid_trading_time'], 'signal'] = 0
    
    # Drop temporary columns
    df.drop(['hour', 'minute', 'time_valid', 'weekday', 'weekday_valid', 'not_holiday', 'valid_trading_time'], axis=1, inplace=True)
    if 'date' in df.columns:
        df.drop('date', axis=1, inplace=True)
    
    return df

# Dictionary of available Indian market strategies
INDIAN_STRATEGIES = {
    "nifty_bank_nifty_momentum": {
        "name": "Nifty/Bank Nifty Momentum",
        "description": "Momentum-based strategy for Nifty and Bank Nifty using EMA crossovers and RSI",
        "function": nifty_bank_nifty_momentum,
        "parameters": {
            "fast_period": {
                "type": "int",
                "default": 9,
                "min": 3,
                "max": 21,
                "description": "Fast EMA period"
            },
            "slow_period": {
                "type": "int",
                "default": 21,
                "min": 10,
                "max": 50,
                "description": "Slow EMA period"
            }
        },
        "markets": ["NSE"],
        "timeframes": ["1D", "1h", "30m"],
        "suitable_for": ["Nifty 50", "Bank Nifty"]
    },
    "nifty_gap_trading": {
        "name": "Nifty Gap Trading",
        "description": "Gap trading strategy for Nifty index based on overnight price gaps",
        "function": nifty_gap_trading,
        "parameters": {},  # No additional parameters
        "markets": ["NSE"],
        "timeframes": ["1D"],
        "suitable_for": ["Nifty 50", "Nifty 500 Stocks"]
    },
    "vwap_nse_intraday": {
        "name": "NSE VWAP Intraday",
        "description": "VWAP-based intraday strategy for NSE stocks with deviation bands",
        "function": vwap_nse_intraday,
        "parameters": {
            "deviation": {
                "type": "float",
                "default": 1.5,
                "min": 0.5,
                "max": 3.0,
                "description": "Standard deviation multiplier for bands"
            }
        },
        "markets": ["NSE"],
        "timeframes": ["5m", "15m", "30m", "1h"],
        "suitable_for": ["Nifty 50 Stocks", "Highly liquid NSE stocks"]
    },
    "option_writing_nifty": {
        "name": "Nifty Option Writing",
        "description": "Option writing strategy for Nifty based on IV and days to expiry",
        "function": option_writing_nifty,
        "parameters": {
            "days_to_expiry": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 30,
                "description": "Days remaining to expiry"
            },
            "iv_threshold": {
                "type": "float",
                "default": 20.0,
                "min": 10.0,
                "max": 40.0,
                "description": "IV threshold for signal generation"
            }
        },
        "markets": ["NSE"],
        "timeframes": ["1D", "1h"],
        "suitable_for": ["Nifty Options", "Bank Nifty Options"]
    },
    "fii_dii_flow_strategy": {
        "name": "FII/DII Flow Strategy",
        "description": "Strategy based on institutional money flow in Indian markets",
        "function": fii_dii_flow_strategy,
        "parameters": {},  # Parameters would be FII/DII data, handled separately
        "markets": ["NSE", "BSE"],
        "timeframes": ["1D"],
        "suitable_for": ["Nifty 50", "Sensex", "Midcap indices"]
    },
    "nifty_bank_nifty_correlation": {
        "name": "Nifty-Bank Nifty Correlation",
        "description": "Trading divergence between Nifty and Bank Nifty indices",
        "function": nifty_bank_nifty_correlation,
        "parameters": {
            "correlation_window": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 50,
                "description": "Window for correlation calculation"
            },
            "threshold": {
                "type": "float",
                "default": 0.7,
                "min": 0.3,
                "max": 0.9,
                "description": "Correlation threshold for signals"
            }
        },
        "markets": ["NSE"],
        "timeframes": ["1D", "1h"],
        "suitable_for": ["Nifty 50", "Bank Nifty"]
    }
}

def get_indian_market_strategies():
    """Return the dictionary of available Indian market strategies"""
    return INDIAN_STRATEGIES

def list_indian_strategies():
    """Return a list of available Indian market strategy names"""
    return list(INDIAN_STRATEGIES.keys())

def get_strategy_details(strategy_name):
    """Return details for a specific Indian market strategy"""
    return INDIAN_STRATEGIES.get(strategy_name, None)

def run_indian_strategy(strategy_name, data, **params):
    """
    Run a specific Indian market strategy on the provided data
    
    Args:
        strategy_name (str): Name of the strategy to run
        data (DataFrame): OHLCV data to run the strategy on
        **params: Strategy-specific parameters
        
    Returns:
        DataFrame: Data with added signal columns
    """
    strategy = INDIAN_STRATEGIES.get(strategy_name)
    if not strategy:
        raise ValueError(f"Strategy '{strategy_name}' not found")
    
    # Get the strategy function
    strategy_func = strategy['function']
    
    # Run the strategy with provided parameters
    return strategy_func(data, **params)