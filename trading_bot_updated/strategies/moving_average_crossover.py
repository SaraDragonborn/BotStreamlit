"""
Moving Average Crossover Strategy
=======================================
Implementation of a Moving Average Crossover strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .strategy_base import StrategyBase
from utils.logger import strategy_logger as logger

class MovingAverageCrossover(StrategyBase):
    """
    Moving Average Crossover trading strategy.
    
    This strategy generates:
    - Buy signal when a faster moving average crosses above a slower moving average
    - Sell signal when a faster moving average crosses below a slower moving average
    """
    
    def __init__(self):
        """Initialize the Moving Average Crossover strategy."""
        super().__init__(
            name="Moving Average Crossover",
            description="Generates signals based on crossovers between fast and slow moving averages."
        )
    
    def _define_parameters(self) -> None:
        """Define the strategy parameters."""
        self.parameters = {
            "fast_ma_period": {
                "type": "int",
                "description": "Fast Moving Average Period",
                "default": 20,
                "min": 2,
                "max": 100
            },
            "slow_ma_period": {
                "type": "int",
                "description": "Slow Moving Average Period",
                "default": 50,
                "min": 5,
                "max": 200
            },
            "ma_type": {
                "type": "str",
                "description": "Moving Average Type",
                "default": "SMA",
                "options": ["SMA", "EMA", "WMA"]
            },
            "price_column": {
                "type": "str",
                "description": "Price column to use for MA calculation",
                "default": "close",
                "options": ["open", "high", "low", "close"]
            }
        }
    
    def generate_signals(self, data: pd.DataFrame, parameters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossovers.
        
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
        fast_period = parameters.get('fast_ma_period', self.parameters['fast_ma_period']['default'])
        slow_period = parameters.get('slow_ma_period', self.parameters['slow_ma_period']['default'])
        ma_type = parameters.get('ma_type', self.parameters['ma_type']['default'])
        price_col = parameters.get('price_column', self.parameters['price_column']['default'])
        
        # Validate parameters
        if not price_col in df.columns:
            logger.warning(f"Price column '{price_col}' not found in data. Using 'close' instead.")
            price_col = 'close'
        
        if fast_period >= slow_period:
            logger.warning("Fast MA period must be less than slow MA period. Adjusting parameters.")
            fast_period = min(fast_period, slow_period - 1)
        
        # Calculate moving averages
        if ma_type == 'SMA':
            df['fast_ma'] = df[price_col].rolling(window=fast_period).mean()
            df['slow_ma'] = df[price_col].rolling(window=slow_period).mean()
        elif ma_type == 'EMA':
            df['fast_ma'] = df[price_col].ewm(span=fast_period, adjust=False).mean()
            df['slow_ma'] = df[price_col].ewm(span=slow_period, adjust=False).mean()
        elif ma_type == 'WMA':
            # Weighted Moving Average
            weights_fast = np.arange(1, fast_period + 1)
            weights_slow = np.arange(1, slow_period + 1)
            
            df['fast_ma'] = df[price_col].rolling(window=fast_period).apply(
                lambda x: np.sum(weights_fast * x) / weights_fast.sum(), raw=True
            )
            df['slow_ma'] = df[price_col].rolling(window=slow_period).apply(
                lambda x: np.sum(weights_slow * x) / weights_slow.sum(), raw=True
            )
        else:
            logger.warning(f"Invalid MA type: {ma_type}. Using SMA instead.")
            df['fast_ma'] = df[price_col].rolling(window=fast_period).mean()
            df['slow_ma'] = df[price_col].rolling(window=slow_period).mean()
        
        # Calculate the difference between fast and slow MAs
        df['ma_diff'] = df['fast_ma'] - df['slow_ma']
        
        # Calculate the previous MA difference
        df['prev_ma_diff'] = df['ma_diff'].shift(1)
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate signals
        # 1 = buy signal (fast MA crosses above slow MA)
        df.loc[(df['ma_diff'] > 0) & (df['prev_ma_diff'] <= 0), 'signal'] = 1
        
        # -1 = sell signal (fast MA crosses below slow MA)
        df.loc[(df['ma_diff'] < 0) & (df['prev_ma_diff'] >= 0), 'signal'] = -1
        
        # Log the number of signals
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()
        logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        return df