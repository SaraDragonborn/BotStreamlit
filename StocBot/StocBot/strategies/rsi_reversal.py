"""
RSI Reversal Strategy Module
=======================================
Strategy that generates signals based on RSI reversals.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Union, Any
from strategies.strategy_base import StrategyBase
import config

class RSIReversal(StrategyBase):
    """
    RSI Reversal Strategy.
    
    Generates buy signals when RSI is oversold and starts to move up with volume
    and sell signals when RSI is overbought and starts to move down with volume.
    
    Attributes:
    -----------
    name : str
        Strategy name
    params : dict
        Strategy parameters including rsi_period, oversold_threshold, overbought_threshold, volume_factor
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the RSI Reversal strategy.
        
        Parameters:
        -----------
        params : dict, optional
            Strategy parameters
        """
        # Get default parameters from config if not provided
        if params is None:
            params = config.get('STRATEGY_PARAMS', {}).get('rsi_reversal', {})
        
        # Default parameters if not in config
        default_params = {
            'rsi_period': 14,      # RSI calculation period 
            'oversold_threshold': 30,  # RSI oversold threshold
            'overbought_threshold': 70, # RSI overbought threshold
            'volume_factor': 1.5,  # Volume increase factor to confirm reversal
        }
        
        # Merge default parameters with provided parameters
        for key, value in default_params.items():
            if key not in params:
                params[key] = value
        
        super().__init__('RSI Reversal', params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI reversals.
        
        Buy signal (1) when RSI < oversold_threshold and starts to move up with volume
        Sell signal (-1) when RSI > overbought_threshold and starts to move down with volume
        
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
        rsi_period = self.params['rsi_period']
        oversold_threshold = self.params['oversold_threshold']
        overbought_threshold = self.params['overbought_threshold']
        volume_factor = self.params['volume_factor']
        
        # Calculate RSI
        signals['rsi'] = ta.momentum.rsi(
            close=signals['close'], 
            window=rsi_period,
            fillna=True
        )
        
        # Calculate average volume
        signals['avg_volume'] = ta.volume.volume_sma_indicator(
            volume=signals['volume'],
            window=5,
            fillna=True
        )
        
        # Volume increase condition
        signals['volume_increase'] = signals['volume'] > signals['avg_volume'] * volume_factor
        
        # Initialize signal column
        signals['signal'] = 0
        
        # RSI reversal conditions
        signals['oversold'] = signals['rsi'] < oversold_threshold
        signals['overbought'] = signals['rsi'] > overbought_threshold
        
        # RSI direction change
        signals['rsi_rising'] = signals['rsi'] > signals['rsi'].shift(1)
        signals['rsi_falling'] = signals['rsi'] < signals['rsi'].shift(1)
        
        # Buy signal when RSI is oversold and starts to rise with increased volume
        buy_condition = (
            signals['oversold'] & 
            signals['rsi_rising'] & 
            signals['volume_increase']
        )
        
        # Sell signal when RSI is overbought and starts to fall with increased volume
        sell_condition = (
            signals['overbought'] & 
            signals['rsi_falling'] & 
            signals['volume_increase']
        )
        
        # Generate signals
        signals['signal'] = np.where(
            buy_condition, 
            1,  # Buy signal
            np.where(
                sell_condition,
                -1,  # Sell signal
                0  # No signal
            )
        )
        
        # Add price change for confirmation
        signals['price_up'] = signals['close'] > signals['close'].shift(1)
        signals['price_down'] = signals['close'] < signals['close'].shift(1)
        
        # Further filter signals based on price movement
        signals['signal'] = np.where(
            (signals['signal'] == 1) & ~signals['price_up'],
            0,  # Remove buy signal if price is not moving up
            np.where(
                (signals['signal'] == -1) & ~signals['price_down'],
                0,  # Remove sell signal if price is not moving down
                signals['signal']
            )
        )
        
        return signals