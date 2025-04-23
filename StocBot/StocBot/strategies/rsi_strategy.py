"""
RSI (Relative Strength Index) Strategy
=======================================
Implementation of a trading strategy based on the Relative Strength Index.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .strategy_base import StrategyBase
from utils.logger import strategy_logger as logger

class RSIStrategy(StrategyBase):
    """
    RSI (Relative Strength Index) trading strategy.
    
    This strategy generates:
    - Buy signal when RSI crosses above the oversold threshold (from below)
    - Sell signal when RSI crosses below the overbought threshold (from above)
    """
    
    def __init__(self):
        """Initialize the RSI strategy."""
        super().__init__(
            name="RSI Strategy",
            description="Generates signals based on RSI overbought and oversold conditions."
        )
    
    def _define_parameters(self) -> None:
        """Define the strategy parameters."""
        self.parameters = {
            "rsi_period": {
                "type": "int",
                "description": "RSI Period",
                "default": 14,
                "min": 2,
                "max": 50
            },
            "overbought_threshold": {
                "type": "int",
                "description": "Overbought Threshold",
                "default": 70,
                "min": 50,
                "max": 90
            },
            "oversold_threshold": {
                "type": "int",
                "description": "Oversold Threshold",
                "default": 30,
                "min": 10,
                "max": 50
            },
            "price_column": {
                "type": "str",
                "description": "Price column to use for RSI calculation",
                "default": "close",
                "options": ["open", "high", "low", "close"]
            }
        }
    
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
        Generate trading signals based on RSI.
        
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
        rsi_period = parameters.get('rsi_period', self.parameters['rsi_period']['default'])
        overbought = parameters.get('overbought_threshold', self.parameters['overbought_threshold']['default'])
        oversold = parameters.get('oversold_threshold', self.parameters['oversold_threshold']['default'])
        price_col = parameters.get('price_column', self.parameters['price_column']['default'])
        
        # Validate parameters
        if not price_col in df.columns:
            logger.warning(f"Price column '{price_col}' not found in data. Using 'close' instead.")
            price_col = 'close'
        
        if oversold >= overbought:
            logger.warning("Oversold threshold must be less than overbought threshold. Adjusting parameters.")
            oversold = min(oversold, overbought - 10)
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df[price_col], rsi_period)
        
        # Calculate previous RSI
        df['prev_rsi'] = df['rsi'].shift(1)
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate signals
        # 1 = buy signal (RSI crosses above oversold threshold)
        df.loc[(df['rsi'] > oversold) & (df['prev_rsi'] <= oversold), 'signal'] = 1
        
        # -1 = sell signal (RSI crosses below overbought threshold)
        df.loc[(df['rsi'] < overbought) & (df['prev_rsi'] >= overbought), 'signal'] = -1
        
        # Log the number of signals
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()
        logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        return df