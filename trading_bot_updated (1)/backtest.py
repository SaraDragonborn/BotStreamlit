"""
Backtesting Module
=======================================
Backtests trading strategies on historical data.
"""

import os
import json
import datetime
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple
from data_collector import data_collector
from strategies.strategy_base import StrategyBase
from strategies.moving_average_crossover import MovingAverageCrossover
from strategies.rsi_strategy import RSIStrategy
from strategies.trend_strategy import TrendFollowingStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.breakout_strategy import BreakoutStrategy
from utils.logger import get_trade_logger
from config import get_config, BACKTEST_CONFIG

config = get_config()
logger = get_trade_logger()

class Backtester:
    """
    Backtester for trading strategies.
    
    Tests strategies on historical data and calculates performance metrics.
    """
    
    def __init__(self, 
                initial_capital: float = 100000.0,
                commission: float = 0.001,  # 0.1%
                slippage: float = 0.001,    # 0.1%
                data_source: str = 'auto'):
        """
        Initialize the Backtester.
        
        Parameters:
        -----------
        initial_capital : float, default=100000.0
            Initial capital for backtesting
        commission : float, default=0.001
            Commission rate as decimal (0.001 = 0.1%)
        slippage : float, default=0.001
            Slippage rate as decimal (0.001 = 0.1%)
        data_source : str, default='auto'
            Data source for historical data
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.data_source = data_source
        
        # Available strategies
        self.strategies = {
            'moving_average_crossover': MovingAverageCrossover(),
            'rsi': RSIStrategy(),
            'trend_following': TrendFollowingStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'breakout': BreakoutStrategy()
        }
        
        logger.info(f"Backtester initialized with {len(self.strategies)} strategies")
    
    def run_backtest(self, 
                   symbol: str,
                   strategy_name: str,
                   start_date: str,
                   end_date: str,
                   timeframe: str = '1d',
                   parameters: Optional[Dict] = None) -> Dict:
        """
        Run a backtest for a specific strategy on a specific symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to backtest
        strategy_name : str
            Name of the strategy to use
        start_date : str
            Start date (format: 'YYYY-MM-DD')
        end_date : str
            End date (format: 'YYYY-MM-DD')
        timeframe : str, default='1d'
            Data timeframe ('1h', '1d')
        parameters : Dict, optional
            Custom strategy parameters
            
        Returns:
        --------
        Dict
            Backtest results
        """
        # Get the strategy
        if strategy_name not in self.strategies:
            logger.error(f"Strategy {strategy_name} not found")
            return {'error': f"Strategy {strategy_name} not found"}
        
        strategy = self.strategies[strategy_name]
        
        # Get historical data
        data = data_collector.get_stock_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=self.data_source
        )
        
        if data.empty:
            logger.error(f"No data found for {symbol} from {start_date} to {end_date}")
            return {'error': f"No data found for {symbol}"}
        
        # Log data summary
        logger.info(f"Running backtest for {symbol} with {strategy_name} strategy")
        logger.info(f"Data summary: {len(data)} bars from {data['datetime'].min()} to {data['datetime'].max()}")
        
        # Add slippage and commission to backtest parameters
        backtest_params = {
            'initial_capital': self.initial_capital,
            'commission': self.commission,
            'slippage': self.slippage
        }
        
        # Run the backtest
        try:
            # Generate signals using strategy
            signals = strategy.generate_signals(data, parameters)
            
            # Add execution price columns with slippage
            signals['buy_price'] = signals['open'] * (1 + self.slippage)
            signals['sell_price'] = signals['open'] * (1 - self.slippage)
            
            # Initialize backtest variables
            capital = self.initial_capital
            position = 0
            trades = []
            equity_curve = [capital]
            daily_returns = []
            max_drawdown = 0
            max_equity = capital
            
            # Simulate trading
            for i in range(1, len(signals)):
                # Current and previous data
                current = signals.iloc[i]
                prev = signals.iloc[i-1]
                
                # Update equity if holding position
                if position > 0:
                    current_equity = capital + (position * current['close'])
                    equity_curve.append(current_equity)
                    
                    # Calculate daily return
                    daily_return = (current_equity / equity_curve[-2]) - 1
                    daily_returns.append(daily_return)
                    
                    # Update max equity and drawdown
                    if current_equity > max_equity:
                        max_equity = current_equity
                    
                    drawdown = 1 - (current_equity / max_equity) if max_equity > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
                else:
                    equity_curve.append(capital)
                    daily_returns.append(0)
                
                # Check for buy signal
                if prev['signal'] == 1 and position == 0:
                    # Enter long position
                    price = current['buy_price']  # Buy at open with slippage
                    shares = int((capital * 0.95) / price)  # Use 95% of capital
                    cost = shares * price
                    commission_cost = cost * self.commission
                    
                    if shares > 0:
                        capital -= (cost + commission_cost)
                        position = shares
                        
                        # Record trade
                        trades.append({
                            'entry_date': current['datetime'] if isinstance(current['datetime'], str) else current['datetime'].isoformat(),
                            'entry_price': price,
                            'shares': shares,
                            'type': 'buy',
                            'exit_date': None,
                            'exit_price': None,
                            'pnl': 0,
                            'pnl_pct': 0,
                            'commission': commission_cost
                        })
                
                # Check for sell signal
                elif prev['signal'] == -1 and position > 0:
                    # Exit long position
                    price = current['sell_price']  # Sell at open with slippage
                    value = position * price
                    commission_cost = value * self.commission
                    
                    capital += (value - commission_cost)
                    
                    # Calculate P&L
                    entry_price = trades[-1]['entry_price']
                    pnl = value - (entry_price * position) - commission_cost - trades[-1]['commission']
                    pnl_pct = (price / entry_price) - 1 - (self.commission * 2)
                    
                    # Update trade record
                    trades[-1]['exit_date'] = current['datetime'] if isinstance(current['datetime'], str) else current['datetime'].isoformat()
                    trades[-1]['exit_price'] = price
                    trades[-1]['pnl'] = pnl
                    trades[-1]['pnl_pct'] = pnl_pct
                    
                    position = 0
            
            # Handle any open position at the end
            if position > 0:
                last_price = signals.iloc[-1]['close']
                value = position * last_price
                commission_cost = value * self.commission
                
                # Calculate P&L
                entry_price = trades[-1]['entry_price']
                pnl = value - (entry_price * position) - commission_cost - trades[-1]['commission']
                pnl_pct = (last_price / entry_price) - 1 - (self.commission * 2)
                
                # Update trade record
                trades[-1]['exit_date'] = signals.iloc[-1]['datetime'] if isinstance(signals.iloc[-1]['datetime'], str) else signals.iloc[-1]['datetime'].isoformat()
                trades[-1]['exit_price'] = last_price
                trades[-1]['pnl'] = pnl
                trades[-1]['pnl_pct'] = pnl_pct
                
                # Update final equity
                capital += (value - commission_cost)
                equity_curve[-1] = capital
                position = 0
            
            # Calculate performance metrics
            total_return = (capital / self.initial_capital) - 1
            
            # Calculate annualized return
            days = (datetime.datetime.strptime(end_date, '%Y-%m-%d') - datetime.datetime.strptime(start_date, '%Y-%m-%d')).days
            years = days / 365
            annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
            
            # Calculate Sharpe ratio (assuming 0% risk-free rate)
            if daily_returns:
                daily_returns_std = np.std(daily_returns)
                sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / daily_returns_std if daily_returns_std > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate trade metrics
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            # Calculate profit factor
            gross_profit = sum(t['pnl'] for t in winning_trades)
            gross_loss = abs(sum(t['pnl'] for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Create benchmark comparison
            try:
                if symbol != 'SPY' and timeframe == '1d':
                    benchmark_data = data_collector.get_stock_data(
                        symbol='SPY',
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        source='yfinance'
                    )
                    
                    if not benchmark_data.empty:
                        first_price = benchmark_data.iloc[0]['close']
                        last_price = benchmark_data.iloc[-1]['close']
                        benchmark_return = (last_price / first_price) - 1
                    else:
                        benchmark_return = 0
                else:
                    benchmark_return = 0
            except Exception as e:
                logger.error(f"Error calculating benchmark return: {str(e)}")
                benchmark_return = 0
            
            # Compile backtest results
            results = {
                'symbol': symbol,
                'strategy': strategy_name,
                'parameters': parameters or {k: v['default'] for k, v in strategy.parameters.items()},
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': self.initial_capital,
                'final_capital': capital,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'annualized_return': annualized_return,
                'annualized_return_pct': annualized_return * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'benchmark_return': benchmark_return,
                'benchmark_return_pct': benchmark_return * 100,
                'outperformance': total_return - benchmark_return,
                'outperformance_pct': (total_return - benchmark_return) * 100,
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'trades': trades
            }
            
            # Save equity curve data
            timestamps = [signals.iloc[i]['datetime'] if isinstance(signals.iloc[i]['datetime'], str) 
                        else signals.iloc[i]['datetime'].isoformat() 
                        for i in range(len(equity_curve))]
            
            results['equity_curve'] = {
                'timestamps': timestamps,
                'equity': equity_curve
            }
            
            logger.info(f"Backtest completed for {symbol} with {strategy_name}")
            logger.info(f"Total return: {results['total_return_pct']:.2f}%, Sharpe: {results['sharpe_ratio']:.2f}, "
                       f"Win rate: {results['win_rate_pct']:.2f}%, Trades: {results['total_trades']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return {'error': f"Error running backtest: {str(e)}"}
    
    def optimize_strategy(self, 
                        symbol: str,
                        strategy_name: str,
                        start_date: str,
                        end_date: str,
                        param_grid: Dict[str, List],
                        timeframe: str = '1d',
                        metric: str = 'sharpe_ratio') -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Parameters:
        -----------
        symbol : str
            Symbol to optimize for
        strategy_name : str
            Name of the strategy to optimize
        start_date : str
            Start date (format: 'YYYY-MM-DD')
        end_date : str
            End date (format: 'YYYY-MM-DD')
        param_grid : Dict[str, List]
            Dictionary of parameter names and values to try
        timeframe : str, default='1d'
            Data timeframe ('1h', '1d')
        metric : str, default='sharpe_ratio'
            Metric to optimize for
            
        Returns:
        --------
        Dict
            Optimization results
        """
        # Get the strategy
        if strategy_name not in self.strategies:
            logger.error(f"Strategy {strategy_name} not found")
            return {'error': f"Strategy {strategy_name} not found"}
        
        strategy = self.strategies[strategy_name]
        
        # Get data once
        data = data_collector.get_stock_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=self.data_source
        )
        
        if data.empty:
            logger.error(f"No data found for {symbol} from {start_date} to {end_date}")
            return {'error': f"No data found for {symbol}"}
        
        # Generate all parameter combinations
        import itertools
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Optimizing {strategy_name} for {symbol} with {len(param_combinations)} parameter combinations")
        
        # Run backtest for each parameter combination
        results = []
        
        for i, combination in enumerate(param_combinations):
            # Create parameter dictionary
            params = {param_names[j]: combination[j] for j in range(len(param_names))}
            
            # Run backtest with these parameters
            result = self.run_backtest(
                symbol=symbol,
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                parameters=params
            )
            
            if 'error' not in result:
                # Store result with parameters
                optimization_result = {
                    'parameters': params,
                    'metric_value': result.get(metric, 0),
                    'backtest_result': result
                }
                
                results.append(optimization_result)
                
                logger.info(f"Optimization progress: {i+1}/{len(param_combinations)} - "
                           f"{metric}: {result.get(metric, 0):.4f}")
        
        # Find best result
        if results:
            best_result = max(results, key=lambda x: x['metric_value'])
        else:
            logger.error("No valid backtest results found")
            return {'error': "No valid backtest results found"}
        
        # Compile optimization results
        optimization_result = {
            'symbol': symbol,
            'strategy': strategy_name,
            'metric': metric,
            'best_parameters': best_result['parameters'],
            'best_metric_value': best_result['metric_value'],
            'best_backtest_result': best_result['backtest_result'],
            'all_results': results
        }
        
        logger.info(f"Optimization completed. Best {metric}: {best_result['metric_value']:.4f}")
        logger.info(f"Best parameters: {best_result['parameters']}")
        
        return optimization_result
    
    def save_results(self, results: Dict, filename: str) -> None:
        """
        Save backtest results to a file.
        
        Parameters:
        -----------
        results : Dict
            Backtest results
        filename : str
            Filename to save results to
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def plot_equity_curve(self, results: Dict, filename: Optional[str] = None) -> None:
        """
        Plot equity curve from backtest results.
        
        Parameters:
        -----------
        results : Dict
            Backtest results
        filename : str, optional
            Filename to save plot to
        """
        try:
            # Set up plot
            plt.figure(figsize=(12, 6))
            
            # Extract equity curve data
            timestamps = results['equity_curve']['timestamps']
            equity = results['equity_curve']['equity']
            
            # Convert timestamps to datetime if they're strings
            if isinstance(timestamps[0], str):
                timestamps = [datetime.datetime.fromisoformat(ts) if 'T' in ts 
                            else datetime.datetime.strptime(ts, '%Y-%m-%d') 
                            for ts in timestamps]
            
            # Plot equity curve
            plt.plot(timestamps, equity, label='Equity Curve')
            
            # Add initial capital reference line
            plt.axhline(y=results['initial_capital'], color='r', linestyle='--', label='Initial Capital')
            
            # Add labels and title
            plt.title(f"{results['strategy']} Backtest on {results['symbol']} ({results['start_date']} to {results['end_date']})")
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.legend()
            
            # Rotate date labels
            plt.gcf().autofmt_xdate()
            
            # Add performance stats
            stats_text = (
                f"Total Return: {results['total_return_pct']:.2f}%\n"
                f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {results['max_drawdown_pct']:.2f}%\n"
                f"Win Rate: {results['win_rate_pct']:.2f}%\n"
                f"Profit Factor: {results['profit_factor']:.2f}"
            )
            plt.figtext(0.15, 0.15, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            # Save or show plot
            if filename:
                plt.savefig(filename)
                logger.info(f"Plot saved to {filename}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")
    
    def run_all_strategies(self, 
                         symbol: str,
                         start_date: str,
                         end_date: str,
                         timeframe: str = '1d') -> Dict:
        """
        Run all available strategies on a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to backtest
        start_date : str
            Start date (format: 'YYYY-MM-DD')
        end_date : str
            End date (format: 'YYYY-MM-DD')
        timeframe : str, default='1d'
            Data timeframe ('1h', '1d')
            
        Returns:
        --------
        Dict
            Results for all strategies
        """
        all_results = {}
        
        for strategy_name in self.strategies:
            logger.info(f"Running {strategy_name} strategy...")
            
            result = self.run_backtest(
                symbol=symbol,
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            all_results[strategy_name] = result
        
        # Rank strategies by Sharpe ratio
        strategy_metrics = [
            {
                'strategy': strategy,
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'total_return': results.get('total_return', 0),
                'win_rate': results.get('win_rate', 0),
                'profit_factor': results.get('profit_factor', 0)
            }
            for strategy, results in all_results.items()
            if 'error' not in results
        ]
        
        ranked_strategies = sorted(strategy_metrics, key=lambda x: x['sharpe_ratio'], reverse=True)
        
        all_results['ranked_strategies'] = ranked_strategies
        all_results['best_strategy'] = ranked_strategies[0]['strategy'] if ranked_strategies else None
        
        return all_results


def main():
    """Run backtests from command line."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    
    parser.add_argument('--strategy', type=str, default='moving_average_crossover',
                        help='Strategy to backtest')
    parser.add_argument('--symbol', type=str, default='SPY',
                        help='Symbol to backtest')
    parser.add_argument('--start', type=str, default='2022-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-01-01',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, default='1d',
                        help='Data timeframe (1d, 1h)')
    parser.add_argument('--capital', type=float, default=100000.0,
                        help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate (decimal)')
    parser.add_argument('--slippage', type=float, default=0.001,
                        help='Slippage rate (decimal)')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize strategy parameters')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all strategies')
    parser.add_argument('--save', type=str, default='',
                        help='Save results to file')
    parser.add_argument('--plot', action='store_true',
                        help='Plot equity curve')
    
    args = parser.parse_args()
    
    # Create backtester
    backtester = Backtester(
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage
    )
    
    if args.compare:
        # Run all strategies
        results = backtester.run_all_strategies(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            timeframe=args.timeframe
        )
        
        # Print ranked strategies
        print("\nStrategy Rankings (by Sharpe Ratio):")
        for i, strategy in enumerate(results['ranked_strategies']):
            print(f"{i+1}. {strategy['strategy']}: Sharpe={strategy['sharpe_ratio']:.2f}, "
                 f"Return={strategy['total_return']*100:.2f}%, Win Rate={strategy['win_rate']*100:.2f}%")
        
        if args.save:
            backtester.save_results(results, args.save)
        
    elif args.optimize:
        # Define parameter grid for the selected strategy
        param_grids = {
            'moving_average_crossover': {
                'fast_ma_period': [10, 20, 50],
                'slow_ma_period': [50, 100, 200],
                'ma_type': ['SMA', 'EMA']
            },
            'rsi': {
                'rsi_period': [5, 9, 14, 21],
                'oversold_threshold': [20, 30, 40],
                'overbought_threshold': [60, 70, 80]
            },
            'trend_following': {
                'ema_period': [10, 21, 50],
                'adx_period': [7, 14, 21],
                'adx_threshold': [20, 25, 30]
            },
            'mean_reversion': {
                'bb_period': [10, 20, 30],
                'bb_std_dev': [1.5, 2.0, 2.5],
                'rsi_period': [7, 14, 21]
            },
            'breakout': {
                'channel_period': [10, 20, 30],
                'volume_factor': [1.2, 1.5, 2.0],
                'atr_multiplier': [0.5, 1.0, 1.5]
            }
        }
        
        # Run optimization
        if args.strategy in param_grids:
            results = backtester.optimize_strategy(
                symbol=args.symbol,
                strategy_name=args.strategy,
                start_date=args.start,
                end_date=args.end,
                param_grid=param_grids[args.strategy],
                timeframe=args.timeframe
            )
            
            # Print best parameters
            print("\nBest Parameters:")
            for param, value in results['best_parameters'].items():
                print(f"{param}: {value}")
            
            # Print performance metrics
            best_result = results['best_backtest_result']
            print(f"\nPerformance Metrics:")
            print(f"Total Return: {best_result['total_return_pct']:.2f}%")
            print(f"Annualized Return: {best_result['annualized_return_pct']:.2f}%")
            print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {best_result['max_drawdown_pct']:.2f}%")
            print(f"Win Rate: {best_result['win_rate_pct']:.2f}%")
            print(f"Profit Factor: {best_result['profit_factor']:.2f}")
            print(f"Total Trades: {best_result['total_trades']}")
            
            if args.save:
                backtester.save_results(results, args.save)
            
            if args.plot:
                plot_file = args.save.replace('.json', '.png') if args.save else None
                backtester.plot_equity_curve(best_result, plot_file)
        else:
            print(f"No parameter grid defined for strategy {args.strategy}")
    
    else:
        # Run single backtest
        results = backtester.run_backtest(
            symbol=args.symbol,
            strategy_name=args.strategy,
            start_date=args.start,
            end_date=args.end,
            timeframe=args.timeframe
        )
        
        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            # Print performance metrics
            print(f"\nPerformance Metrics:")
            print(f"Total Return: {results['total_return_pct']:.2f}%")
            print(f"Annualized Return: {results['annualized_return_pct']:.2f}%")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
            print(f"Win Rate: {results['win_rate_pct']:.2f}%")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Benchmark Return: {results['benchmark_return_pct']:.2f}%")
            print(f"Outperformance: {results['outperformance_pct']:.2f}%")
            
            if args.save:
                backtester.save_results(results, args.save)
            
            if args.plot:
                plot_file = args.save.replace('.json', '.png') if args.save else None
                backtester.plot_equity_curve(results, plot_file)


if __name__ == '__main__':
    main()