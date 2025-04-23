"""
Moving Average Crossover Strategy Module
=======================================
Strategy that generates signals based on EMA crossovers.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Union, Any
from strategies.strategy_base import StrategyBase
import config

class MovingAverageCrossover(StrategyBase):
    """
    Moving Average Crossover Strategy.
    
    Generates buy signals when short-term EMA crosses above long-term EMA
    and sell signals when short-term EMA crosses below long-term EMA.
    
    Attributes:
    -----------
    name : str
        Strategy name
    params : dict
        Strategy parameters including short_window, long_window, use_ema
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the Moving Average Crossover strategy.
        
        Parameters:
        -----------
        params : dict, optional
            Strategy parameters
        """
        # Get default parameters from config if not provided
        if params is None:
            params = config.get('STRATEGY_PARAMS', {}).get('moving_average_crossover', {})
        
        # Default parameters if not in config
        default_params = {
            'short_window': 15,    # Fast EMA (15 period)
            'long_window': 50,     # Slow EMA (50 period)
            'use_ema': True,       # Use EMA instead of SMA
        }
        
        # Merge default parameters with provided parameters
        for key, value in default_params.items():
            if key not in params:
                params[key] = value
        
        super().__init__('Moving Average Crossover', params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on EMA crossovers.
        
        Buy signal (1) when short-term EMA crosses above long-term EMA
        Sell signal (-1) when short-term EMA crosses below long-term EMA
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data with OHLCV columns
            
        Returns:
        --------
        pandas.DataFrame
            Price data with added signal column
        """
        if data is None or data.empty:
            return pd.DataFrame()
        
        # Make a copy of the data to avoid modifying the original
        signals = data.copy()
        
        # Get parameters
        short_window = self.params['short_window']
        long_window = self.params['long_window']
        use_ema = self.params['use_ema']
        
        # Calculate moving averages
        if use_ema:
            # Calculate EMAs
            signals['short_ma'] = ta.trend.ema_indicator(
                close=signals['close'], 
                window=short_window,
                fillna=True
            )
            signals['long_ma'] = ta.trend.ema_indicator(
                close=signals['close'], 
                window=long_window,
                fillna=True
            )
        else:
            # Calculate SMAs
            signals['short_ma'] = ta.trend.sma_indicator(
                close=signals['close'], 
                window=short_window,
                fillna=True
            )
            signals['long_ma'] = ta.trend.sma_indicator(
                close=signals['close'], 
                window=long_window,
                fillna=True
            )
        
        # Initialize signal column
        signals['signal'] = 0
        
        # Generate signals
        # Buy signal (1) when short-term MA crosses above long-term MA
        # Sell signal (-1) when short-term MA crosses below long-term MA
        signals['signal'] = np.where(
            (signals['short_ma'] > signals['long_ma']) & 
            (signals['short_ma'].shift(1) <= signals['long_ma'].shift(1)),
            1,  # Buy signal
            np.where(
                (signals['short_ma'] < signals['long_ma']) & 
                (signals['short_ma'].shift(1) >= signals['long_ma'].shift(1)),
                -1,  # Sell signal
                0  # No signal
            )
        )
        
        return signals