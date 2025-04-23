"""
Mean Reversion Strategy
=======================================
Implementation of a Mean Reversion strategy using Bollinger Bands and RSI.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .strategy_base import StrategyBase
from utils.logger import strategy_logger as logger

class MeanReversionStrategy(StrategyBase):
    """
    Mean Reversion trading strategy.
    
    This strategy generates:
    - Buy signal when price touches lower Bollinger Band and RSI is oversold
    - Sell signal when price touches upper Bollinger Band and RSI is overbought
    """
    
    def __init__(self):
        """Initialize the Mean Reversion strategy."""
        super().__init__(
            name="Mean Reversion",
            description="Generates signals based on price reverting to mean using Bollinger Bands and RSI."
        )
    
    def _define_parameters(self) -> None:
        """Define the strategy parameters."""
        self.parameters = {
            "bb_period": {
                "type": "int",
                "description": "Bollinger Bands Period",
                "default": 20,
                "min": 5,
                "max": 50
            },
            "bb_std_dev": {
                "type": "float",
                "description": "Bollinger Bands Standard Deviation",
                "default": 2.0,
                "min": 1.0,
                "max": 3.0
            },
            "rsi_period": {
                "type": "int",
                "description": "RSI Period",
                "default": 14,
                "min": 5,
                "max": 30
            },
            "rsi_oversold": {
                "type": "int",
                "description": "RSI Oversold Threshold",
                "default": 30,
                "min": 10,
                "max": 40
            },
            "rsi_overbought": {
                "type": "int",
                "description": "RSI Overbought Threshold",
                "default": 70,
                "min": 60,
                "max": 90
            },
            "price_column": {
                "type": "str",
                "description": "Price column to use for calculations",
                "default": "close",
                "options": ["open", "high", "low", "close"]
            }
        }
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, price_col: str, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data
        price_col : str
            Price column to use
        period : int, default=20
            Moving average period
        num_std : float, default=2.0
            Number of standard deviations
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Bollinger Bands
        """
        # Make a copy of the data
        df = data.copy()
        
        # Calculate middle band (Simple Moving Average)
        df['bb_middle'] = df[price_col].rolling(window=period).mean()
        
        # Calculate standard deviation
        df['bb_std_dev'] = df[price_col].rolling(window=period).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (df['bb_std_dev'] * num_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std_dev'] * num_std)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index.
        
        Parameters:
        -----------
        prices : pandas.Series
            Price series
        period : int, default=14
            RSI period
            
        Returns:
        --------
        pandas.Series
            RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate the average gain and loss
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate the Relative Strength (RS)
        rs = avg_gain / avg_loss
        
        # Calculate the RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: pd.DataFrame, parameters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate trading signals based on mean reversion strategy.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data with columns: open, high, low, close, volume
        parameters : Dict, optional
            Strategy parameters (if None, use default parameters)
            
        Returns:
        --------
        pandas.DataFrame
            Data with signal column added
        """
        # Use default parameters if none provided
        if parameters is None:
            parameters = {k: v['default'] for k, v in self.parameters.items()}
        
        # Copy the data to avoid modifying the original
        df = data.copy()
        
        # Extract parameters
        bb_period = parameters.get('bb_period', self.parameters['bb_period']['default'])
        bb_std_dev = parameters.get('bb_std_dev', self.parameters['bb_std_dev']['default'])
        rsi_period = parameters.get('rsi_period', self.parameters['rsi_period']['default'])
        rsi_oversold = parameters.get('rsi_oversold', self.parameters['rsi_oversold']['default'])
        rsi_overbought = parameters.get('rsi_overbought', self.parameters['rsi_overbought']['default'])
        price_col = parameters.get('price_column', self.parameters['price_column']['default'])
        
        # Validate parameters
        if not price_col in df.columns:
            logger.warning(f"Price column '{price_col}' not found in data. Using 'close' instead.")
            price_col = 'close'
        
        # Calculate Bollinger Bands
        df = self._calculate_bollinger_bands(df, price_col, bb_period, bb_std_dev)
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df[price_col], rsi_period)
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate buy signals
        # Price touches or crosses below lower Bollinger Band and RSI is oversold
        df.loc[(df[price_col] <= df['bb_lower']) & (df['rsi'] <= rsi_oversold), 'signal'] = 1
        
        # Generate sell signals
        # Price touches or crosses above upper Bollinger Band and RSI is overbought
        df.loc[(df[price_col] >= df['bb_upper']) & (df['rsi'] >= rsi_overbought), 'signal'] = -1
        
        # Log the number of signals
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()
        logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        return df