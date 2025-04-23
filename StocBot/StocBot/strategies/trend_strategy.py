"""
Trend Following Strategy
=======================================
Implementation of a Trend Following strategy using exponential moving averages.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .strategy_base import StrategyBase
from utils.logger import strategy_logger as logger

class TrendFollowingStrategy(StrategyBase):
    """
    Trend Following trading strategy.
    
    This strategy generates:
    - Buy signal when price rises above EMA and ADX indicates strong trend
    - Sell signal when price falls below EMA and ADX indicates strong trend
    """
    
    def __init__(self):
        """Initialize the Trend Following strategy."""
        super().__init__(
            name="Trend Following",
            description="Generates signals based on price trends and EMA crossovers with ADX filter."
        )
    
    def _define_parameters(self) -> None:
        """Define the strategy parameters."""
        self.parameters = {
            "ema_period": {
                "type": "int",
                "description": "EMA Period",
                "default": 21,
                "min": 5,
                "max": 100
            },
            "adx_period": {
                "type": "int",
                "description": "ADX Period",
                "default": 14,
                "min": 5,
                "max": 50
            },
            "adx_threshold": {
                "type": "int",
                "description": "ADX Threshold for Strong Trend",
                "default": 25,
                "min": 15,
                "max": 40
            },
            "price_column": {
                "type": "str",
                "description": "Price column to use for calculations",
                "default": "close",
                "options": ["open", "high", "low", "close"]
            }
        }
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate the Average Directional Index (ADX).
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data with columns: high, low, close
        period : int, default=14
            ADX period
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with ADX values
        """
        # Make a copy of the data
        df = data.copy()
        
        # True Range
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Directional Movement
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        
        # Positive and Negative Directional Movement
        df['plus_dm'] = 0
        df.loc[(df['up_move'] > df['down_move']) & (df['up_move'] > 0), 'plus_dm'] = df['up_move']
        
        df['minus_dm'] = 0
        df.loc[(df['down_move'] > df['up_move']) & (df['down_move'] > 0), 'minus_dm'] = df['down_move']
        
        # Smoothed Directional Indicators
        # First value
        df['tr_period'] = df['tr'].rolling(window=period).sum()
        df['plus_di_period'] = df['plus_dm'].rolling(window=period).sum()
        df['minus_di_period'] = df['minus_dm'].rolling(window=period).sum()
        
        # DIplus and DIminus
        df['plus_di'] = 100 * (df['plus_di_period'] / df['tr_period'])
        df['minus_di'] = 100 * (df['minus_di_period'] / df['tr_period'])
        
        # Directional Movement Index
        df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
        
        # Average Directional Index
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        return df
    
    def generate_signals(self, data: pd.DataFrame, parameters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate trading signals based on trend following strategy.
        
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
        
        # Check if required columns exist
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in data")
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Extract parameters
        ema_period = parameters.get('ema_period', self.parameters['ema_period']['default'])
        adx_period = parameters.get('adx_period', self.parameters['adx_period']['default'])
        adx_threshold = parameters.get('adx_threshold', self.parameters['adx_threshold']['default'])
        price_col = parameters.get('price_column', self.parameters['price_column']['default'])
        
        # Validate parameters
        if not price_col in df.columns:
            logger.warning(f"Price column '{price_col}' not found in data. Using 'close' instead.")
            price_col = 'close'
        
        # Calculate EMA
        df['ema'] = df[price_col].ewm(span=ema_period, adjust=False).mean()
        
        # Calculate ADX
        df = self._calculate_adx(df, adx_period)
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate signals
        # 1 = buy signal (price above EMA and ADX > threshold)
        df.loc[(df[price_col] > df['ema']) & 
               (df[price_col].shift(1) <= df['ema'].shift(1)) & 
               (df['adx'] > adx_threshold), 'signal'] = 1
        
        # -1 = sell signal (price below EMA and ADX > threshold)
        df.loc[(df[price_col] < df['ema']) & 
               (df[price_col].shift(1) >= df['ema'].shift(1)) & 
               (df['adx'] > adx_threshold), 'signal'] = -1
        
        # Log the number of signals
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()
        logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        return df