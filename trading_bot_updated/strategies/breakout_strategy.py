"""
Breakout Strategy
=======================================
Implementation of a Breakout strategy using price channels and volume confirmation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .strategy_base import StrategyBase
from utils.logger import strategy_logger as logger

class BreakoutStrategy(StrategyBase):
    """
    Breakout trading strategy.
    
    This strategy generates:
    - Buy signal when price breaks above resistance with volume confirmation
    - Sell signal when price breaks below support with volume confirmation
    """
    
    def __init__(self):
        """Initialize the Breakout strategy."""
        super().__init__(
            name="Breakout",
            description="Generates signals based on price breaking out of support/resistance levels with volume confirmation."
        )
    
    def _define_parameters(self) -> None:
        """Define the strategy parameters."""
        self.parameters = {
            "channel_period": {
                "type": "int",
                "description": "Channel Period (for high/low)",
                "default": 20,
                "min": 5,
                "max": 100
            },
            "volume_factor": {
                "type": "float",
                "description": "Volume Factor (multiple of average volume)",
                "default": 1.5,
                "min": 1.0,
                "max": 3.0
            },
            "lookback_periods": {
                "type": "int",
                "description": "Number of periods to confirm a breakout",
                "default": 2,
                "min": 1,
                "max": 5
            },
            "atr_period": {
                "type": "int",
                "description": "ATR Period for volatility filter",
                "default": 14,
                "min": 5,
                "max": 30
            },
            "atr_multiplier": {
                "type": "float",
                "description": "ATR Multiplier for breakout threshold",
                "default": 1.0,
                "min": 0.5,
                "max": 2.0
            }
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data with high, low, close columns
        period : int, default=14
            ATR period
            
        Returns:
        --------
        pandas.Series
            ATR values
        """
        # Make a copy of the data
        df = data.copy()
        
        # Calculate True Range
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Calculate ATR
        atr = df['tr'].rolling(window=period).mean()
        
        return atr
    
    def generate_signals(self, data: pd.DataFrame, parameters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate trading signals based on breakout strategy.
        
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
        required_columns = ['high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in data")
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Extract parameters
        channel_period = parameters.get('channel_period', self.parameters['channel_period']['default'])
        volume_factor = parameters.get('volume_factor', self.parameters['volume_factor']['default'])
        lookback_periods = parameters.get('lookback_periods', self.parameters['lookback_periods']['default'])
        atr_period = parameters.get('atr_period', self.parameters['atr_period']['default'])
        atr_multiplier = parameters.get('atr_multiplier', self.parameters['atr_multiplier']['default'])
        
        # Calculate resistance (highest high over period)
        df['resistance'] = df['high'].rolling(window=channel_period).max()
        
        # Calculate support (lowest low over period)
        df['support'] = df['low'].rolling(window=channel_period).min()
        
        # Calculate average volume
        df['avg_volume'] = df['volume'].rolling(window=channel_period).mean()
        
        # Volume threshold for confirmation
        df['volume_threshold'] = df['avg_volume'] * volume_factor
        
        # Calculate ATR for volatility filter
        df['atr'] = self._calculate_atr(df, atr_period)
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate signals
        for i in range(lookback_periods + 1, len(df)):
            # Get the current row
            current = df.iloc[i]
            
            # Check for resistance breakout (buy signal)
            prev_resistance = df.iloc[i-1]['resistance']
            resistance_breakout = (
                current['close'] > (prev_resistance + (current['atr'] * atr_multiplier)) and  # Price breaks above resistance
                current['volume'] > current['volume_threshold']  # Volume confirmation
            )
            
            # Check for support breakout (sell signal)
            prev_support = df.iloc[i-1]['support']
            support_breakout = (
                current['close'] < (prev_support - (current['atr'] * atr_multiplier)) and  # Price breaks below support
                current['volume'] > current['volume_threshold']  # Volume confirmation
            )
            
            # Set signals
            if resistance_breakout:
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif support_breakout:
                df.iloc[i, df.columns.get_loc('signal')] = -1
        
        # Log the number of signals
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()
        logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        return df