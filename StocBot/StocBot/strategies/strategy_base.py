"""
Base Strategy Module
=======================================
Base class for all trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod

class StrategyBase(ABC):
    """
    Base class for all trading strategies.
    
    Attributes:
    -----------
    name : str
        Strategy name
    params : dict
        Strategy parameters
    """
    
    def __init__(self, name: str, params: Optional[Dict] = None):
        """
        Initialize the strategy.
        
        Parameters:
        -----------
        name : str
            Strategy name
        params : dict, optional
            Strategy parameters
        """
        self.name = name
        self.params = params or {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data with OHLCV columns
            
        Returns:
        --------
        pandas.DataFrame
            Price data with added signal column
        """
        pass
    
    def get_latest_signal(self, data: pd.DataFrame) -> Optional[str]:
        """
        Get the latest trading signal.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data with signals
            
        Returns:
        --------
        str or None
            Latest signal ('BUY', 'SELL', or None)
        """
        if data is None or data.empty:
            return None
        
        # Process the data
        signals = self.generate_signals(data)
        
        # Get the last signal
        if 'signal' in signals.columns:
            latest_signal = signals['signal'].iloc[-1]
            if latest_signal == 1:
                return 'BUY'
            elif latest_signal == -1:
                return 'SELL'
        
        return None
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """
        Backtest the strategy.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data with OHLCV columns
        initial_capital : float, default=100000
            Initial capital for backtesting
            
        Returns:
        --------
        dict
            Backtest results
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        # Create a position column (1 for long, -1 for short, 0 for no position)
        signals['position'] = signals['signal'].fillna(0)
        
        # Create a position change column
        signals['position_change'] = signals['position'].diff()
        
        # Create a price column (use close price)
        signals['price'] = signals['close']
        
        # Calculate returns
        signals['returns'] = signals['price'].pct_change()
        
        # Calculate strategy returns
        signals['strategy_returns'] = signals['position'].shift(1) * signals['returns']
        
        # Calculate cumulative returns
        signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()
        
        # Calculate equity curve
        signals['equity'] = initial_capital * signals['cumulative_returns']
        
        # Calculate drawdown
        signals['peak'] = signals['equity'].cummax()
        signals['drawdown'] = (signals['equity'] - signals['peak']) / signals['peak']
        
        # Calculate buy and sell points
        buy_signals = signals[signals['position_change'] > 0]
        sell_signals = signals[signals['position_change'] < 0]
        
        # Calculate profitability metrics
        total_trades = len(buy_signals) + len(sell_signals)
        profitable_trades = len(signals[signals['strategy_returns'] > 0])
        total_return = signals['equity'].iloc[-1] / initial_capital - 1 if len(signals) > 0 else 0
        max_drawdown = signals['drawdown'].min()
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Create result dictionary
        result = {
            'strategy': self.name,
            'params': self.params,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': np.mean(signals['strategy_returns']) / np.std(signals['strategy_returns']) if np.std(signals['strategy_returns']) > 0 else 0,
            'equity_curve': signals['equity'].tolist(),
            'buy_points': list(zip(buy_signals.index.tolist(), buy_signals['price'].tolist())),
            'sell_points': list(zip(sell_signals.index.tolist(), sell_signals['price'].tolist()))
        }
        
        return result