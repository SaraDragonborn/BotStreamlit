"""
Strategy Base Class
=======================================
Base class for all trading strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class StrategyBase(ABC):
    """
    Base class for all trading strategies.
    
    Attributes:
    -----------
    name : str
        Strategy name
    description : str
        Strategy description
    parameters : Dict
        Strategy parameters with their descriptions and default values
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize the strategy.
        
        Parameters:
        -----------
        name : str
            Strategy name
        description : str
            Strategy description
        """
        self.name = name
        self.description = description
        self.parameters = {}
        
        # Initialize strategy parameters
        self._define_parameters()
    
    @abstractmethod
    def _define_parameters(self) -> None:
        """Define the strategy parameters."""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, parameters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate trading signals for the given data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data
        parameters : Dict, optional
            Strategy parameters (if None, use default parameters)
            
        Returns:
        --------
        pandas.DataFrame
            Data with signal column added
        """
        pass
    
    def backtest(self, data: pd.DataFrame, parameters: Optional[Dict] = None, initial_capital: float = 10000.0) -> Dict:
        """
        Backtest the strategy on historical data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data
        parameters : Dict, optional
            Strategy parameters (if None, use default parameters)
        initial_capital : float, default=10000.0
            Initial capital for the backtest
            
        Returns:
        --------
        Dict
            Backtest results
        """
        # Use default parameters if none provided
        if parameters is None:
            parameters = {k: v['default'] for k, v in self.parameters.items()}
        
        # Generate signals
        signals = self.generate_signals(data, parameters)
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = [capital]
        
        # Simulate trading
        for i in range(1, len(signals)):
            # Current and previous data
            current = signals.iloc[i]
            prev = signals.iloc[i-1]
            
            # Check for buy signal
            if prev['signal'] == 1 and position == 0:
                # Enter long position
                price = current['open']  # Buy at open
                shares = int(capital / price)
                cost = shares * price
                
                if shares > 0:
                    capital -= cost
                    position = shares
                    
                    # Record trade
                    trades.append({
                        'entry_date': current.name,
                        'entry_price': price,
                        'shares': shares,
                        'type': 'buy',
                        'exit_date': None,
                        'exit_price': None,
                        'pnl': 0,
                        'pnl_pct': 0
                    })
            
            # Check for sell signal
            elif prev['signal'] == -1 and position > 0:
                # Exit long position
                price = current['open']  # Sell at open
                value = position * price
                pnl = value - (trades[-1]['entry_price'] * position)
                pnl_pct = (price / trades[-1]['entry_price']) - 1
                
                capital += value
                position = 0
                
                # Update trade record
                trades[-1]['exit_date'] = current.name
                trades[-1]['exit_price'] = price
                trades[-1]['pnl'] = pnl
                trades[-1]['pnl_pct'] = pnl_pct
            
            # Update equity curve
            equity = capital + (position * current['close'])
            equity_curve.append(equity)
        
        # Handle any open position at the end
        if position > 0:
            last_price = signals.iloc[-1]['close']
            value = position * last_price
            pnl = value - (trades[-1]['entry_price'] * position)
            pnl_pct = (last_price / trades[-1]['entry_price']) - 1
            
            # Update trade record
            trades[-1]['exit_date'] = signals.index[-1]
            trades[-1]['exit_price'] = last_price
            trades[-1]['pnl'] = pnl
            trades[-1]['pnl_pct'] = pnl_pct
            
            # Update final equity
            capital += value
            position = 0
            equity_curve[-1] = capital
        
        # Calculate performance metrics
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(equity_curve)):
            daily_return = (equity_curve[i] / equity_curve[i-1]) - 1
            daily_returns.append(daily_return)
        
        # Calculate annualized return (assuming 252 trading days per year)
        if len(daily_returns) > 0:
            annualized_return = ((1 + total_return) ** (252 / len(daily_returns))) - 1
        else:
            annualized_return = 0
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe_ratio = 0
        
        # Calculate drawdown
        max_equity = equity_curve[0]
        drawdown = 0
        max_drawdown = 0
        drawdown_series = []
        
        for equity in equity_curve:
            max_equity = max(max_equity, equity)
            drawdown = 1 - (equity / max_equity) if max_equity > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
            drawdown_series.append(drawdown)
        
        # Calculate trade metrics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        avg_win_pct = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss_pct = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
        
        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Compile backtest results
        results = {
            'strategy': self.name,
            'parameters': parameters,
            'initial_capital': initial_capital,
            'final_capital': equity_curve[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'equity_curve': equity_curve,
            'drawdown_series': drawdown_series,
            'trades': trades
        }
        
        return results
    
    def optimize(self, data: pd.DataFrame, param_grid: Dict[str, List[Any]], metric: str = 'total_return') -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data
        param_grid : Dict[str, List[Any]]
            Dictionary of parameter names and values to try
        metric : str, default='total_return'
            Metric to optimize
            
        Returns:
        --------
        Dict
            Optimization results with best parameters
        """
        # Generate all parameter combinations
        import itertools
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Run backtest for each parameter combination
        results = []
        
        for combination in param_combinations:
            # Create parameter dictionary
            params = {param_names[i]: combination[i] for i in range(len(param_names))}
            
            # Run backtest
            backtest_result = self.backtest(data, params)
            
            # Store result with parameters
            result = {
                'parameters': params,
                'metric_value': backtest_result.get(metric, 0),
                'backtest_result': backtest_result
            }
            
            results.append(result)
        
        # Find best result
        if results:
            best_result = max(results, key=lambda x: x['metric_value'])
        else:
            best_result = {
                'parameters': {},
                'metric_value': 0,
                'backtest_result': {}
            }
        
        # Compile optimization results
        optimization_result = {
            'strategy': self.name,
            'metric': metric,
            'best_parameters': best_result['parameters'],
            'best_metric_value': best_result['metric_value'],
            'best_backtest_result': best_result['backtest_result'],
            'all_results': results
        }
        
        return optimization_result