"""
Backtesting Module
=======================================
Backtests trading strategies on historical data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import json
from api.angel_api import AngelOneAPI
from strategies import MovingAverageCrossover, RSIReversal, MarketConditionAnalyzer
from utils.logger import setup_logger

# Set up logger
logger = setup_logger('backtest')

class Backtester:
    """
    Backtester for trading strategies.
    
    Tests strategies on historical data and calculates performance metrics.
    
    Attributes:
    -----------
    api : AngelOneAPI
        API connector for Angel One
    initial_capital : float
        Initial capital for backtesting
    commission : float
        Commission rate as decimal
    slippage : float
        Slippage rate as decimal
    """
    
    def __init__(self, 
                initial_capital: float = 100000.0,
                commission: float = 0.001,  # 0.1%
                slippage: float = 0.001):   # 0.1%
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
        """
        self.api = AngelOneAPI()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Create strategies
        self.strategies = {
            'moving_average_crossover': MovingAverageCrossover(),
            'rsi_reversal': RSIReversal()
        }
        
        # Market condition analyzer
        self.market_analyzer = MarketConditionAnalyzer()
        
        logger.info(f"Backtester initialized with {initial_capital} capital")
    
    def get_historical_data(self, symbol: str, timeframe: str = '5minute',
                          start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Get historical data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        timeframe : str, default='5minute'
            Data timeframe
        start_date : str, optional
            Start date (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date (format: 'YYYY-MM-DD')
            
        Returns:
        --------
        pandas.DataFrame or None
            Historical price data or None if error
        """
        # Authenticate with Angel One API if not already
        if not self.api._authenticated:
            if not self.api.authenticate():
                logger.error("Failed to authenticate with Angel One API")
                return None
        
        # Default dates if not provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Get historical data
            data = self.api.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                from_date=start_date,
                to_date=end_date
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return None
    
    def run_backtest(self, symbol: str, strategy_name: str,
                   start_date: str, end_date: str,
                   timeframe: str = '5minute') -> Dict:
        """
        Run a backtest for a specific strategy on a specific symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        strategy_name : str
            Strategy name
        start_date : str
            Start date (format: 'YYYY-MM-DD')
        end_date : str
            End date (format: 'YYYY-MM-DD')
        timeframe : str, default='5minute'
            Data timeframe
            
        Returns:
        --------
        dict
            Backtest results
        """
        # Get strategy
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            logger.error(f"Strategy {strategy_name} not found")
            return {'error': f"Strategy {strategy_name} not found"}
        
        # Get historical data
        data = self.get_historical_data(symbol, timeframe, start_date, end_date)
        if data is None:
            return {'error': f"Failed to get historical data for {symbol}"}
        
        # Run backtest
        results = strategy.backtest(data, self.initial_capital)
        
        # Add symbol and dates to results
        results['symbol'] = symbol
        results['start_date'] = start_date
        results['end_date'] = end_date
        results['timeframe'] = timeframe
        
        # Calculate additional metrics
        results['commission'] = self.commission
        results['slippage'] = self.slippage
        
        # Apply commission and slippage
        net_return = results['total_return'] - (results['total_trades'] * self.commission * 2)  # * 2 for buy and sell
        results['net_return'] = net_return
        
        # Calculate annualized return
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end_date_obj - start_date_obj).days
        if days > 0:
            annualized_return = ((1 + net_return) ** (365 / days)) - 1
            results['annualized_return'] = annualized_return
        else:
            results['annualized_return'] = 0
        
        logger.info(f"Backtest completed for {symbol} with {strategy_name}")
        logger.info(f"Net return: {net_return:.2%}, Win rate: {results['win_rate']:.2%}")
        
        return results
    
    def run_strategy_comparison(self, symbol: str, start_date: str, end_date: str,
                              timeframe: str = '5minute') -> Dict:
        """
        Run a comparison of all strategies on a specific symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        start_date : str
            Start date (format: 'YYYY-MM-DD')
        end_date : str
            End date (format: 'YYYY-MM-DD')
        timeframe : str, default='5minute'
            Data timeframe
            
        Returns:
        --------
        dict
            Comparison results
        """
        results = {}
        
        for strategy_name in self.strategies:
            results[strategy_name] = self.run_backtest(
                symbol=symbol,
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
        
        # Find the best strategy
        best_strategy = max(results, key=lambda x: results[x].get('net_return', -float('inf')))
        results['best_strategy'] = best_strategy
        
        return results
    
    def run_adaptive_strategy_backtest(self, symbol: str, start_date: str, end_date: str,
                                    timeframe: str = '5minute') -> Dict:
        """
        Run a backtest with adaptive strategy selection based on market conditions.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        start_date : str
            Start date (format: 'YYYY-MM-DD')
        end_date : str
            End date (format: 'YYYY-MM-DD')
        timeframe : str, default='5minute'
            Data timeframe
            
        Returns:
        --------
        dict
            Backtest results
        """
        # Get index data for market condition analysis
        index_symbol = self.market_analyzer.params['index_symbol']
        index_data = self.get_historical_data(index_symbol, timeframe, start_date, end_date)
        
        if index_data is None:
            return {'error': f"Failed to get historical data for index {index_symbol}"}
        
        # Get symbol data
        data = self.get_historical_data(symbol, timeframe, start_date, end_date)
        if data is None:
            return {'error': f"Failed to get historical data for {symbol}"}
        
        # Initialize result DataFrame
        signals = data.copy()
        signals['signal'] = 0
        signals['strategy'] = None
        
        # Get all dates in the data
        dates = signals.index.date.unique()
        
        # Process each day
        for date in dates:
            # Get data for this day
            day_index_data = index_data[index_data.index.date == date]
            day_symbol_data = signals[signals.index.date == date]
            
            if day_index_data.empty or day_symbol_data.empty:
                continue
            
            # Analyze market condition for this day
            condition = self.market_analyzer.analyze(day_index_data)
            
            # Select strategy based on market condition
            if condition == 'trending':
                strategy_name = 'moving_average_crossover'
            else:
                strategy_name = 'rsi_reversal'
            
            # Get strategy
            strategy = self.strategies.get(strategy_name)
            
            # Generate signals for this day
            day_signals = strategy.generate_signals(day_symbol_data)
            
            # Update signals DataFrame
            signals.loc[day_signals.index, 'signal'] = day_signals['signal']
            signals.loc[day_signals.index, 'strategy'] = strategy_name
        
        # Calculate positions
        signals['position'] = signals['signal'].fillna(0)
        signals['position_change'] = signals['position'].diff()
        
        # Calculate returns
        signals['price'] = signals['close']
        signals['returns'] = signals['price'].pct_change()
        signals['strategy_returns'] = signals['position'].shift(1) * signals['returns']
        
        # Calculate cumulative returns
        signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()
        
        # Calculate equity curve
        signals['equity'] = self.initial_capital * signals['cumulative_returns']
        
        # Calculate drawdown
        signals['peak'] = signals['equity'].cummax()
        signals['drawdown'] = (signals['equity'] - signals['peak']) / signals['peak']
        
        # Calculate buy and sell points
        buy_signals = signals[signals['position_change'] > 0]
        sell_signals = signals[signals['position_change'] < 0]
        
        # Calculate profitability metrics
        total_trades = len(buy_signals) + len(sell_signals)
        profitable_trades = len(signals[signals['strategy_returns'] > 0])
        total_return = signals['equity'].iloc[-1] / self.initial_capital - 1 if len(signals) > 0 else 0
        max_drawdown = signals['drawdown'].min()
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Apply commission and slippage
        net_return = total_return - (total_trades * self.commission * 2)  # * 2 for buy and sell
        
        # Calculate annualized return
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end_date_obj - start_date_obj).days
        if days > 0:
            annualized_return = ((1 + net_return) ** (365 / days)) - 1
        else:
            annualized_return = 0
        
        # Strategy usage statistics
        strategy_usage = signals['strategy'].value_counts().to_dict()
        
        # Create result dictionary
        result = {
            'strategy': 'adaptive',
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'timeframe': timeframe,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'net_return': net_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': np.mean(signals['strategy_returns']) / np.std(signals['strategy_returns']) if np.std(signals['strategy_returns']) > 0 else 0,
            'strategy_usage': strategy_usage,
            'equity_curve': signals['equity'].tolist(),
            'buy_points': list(zip(buy_signals.index.astype(str).tolist(), buy_signals['price'].tolist())),
            'sell_points': list(zip(sell_signals.index.astype(str).tolist(), sell_signals['price'].tolist()))
        }
        
        logger.info(f"Adaptive strategy backtest completed for {symbol}")
        logger.info(f"Net return: {net_return:.2%}, Win rate: {win_rate:.2%}")
        
        return result
    
    def save_results(self, results: Dict, filename: str) -> bool:
        """
        Save backtest results to a file.
        
        Parameters:
        -----------
        results : dict
            Backtest results
        filename : str
            Filename to save results to
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Convert datetime objects to strings
            def json_serializer(obj):
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            # Save results to file
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4, default=json_serializer)
            
            logger.info(f"Results saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False
    
    def plot_equity_curve(self, results: Dict, filename: Optional[str] = None) -> None:
        """
        Plot equity curve from backtest results.
        
        Parameters:
        -----------
        results : dict
            Backtest results
        filename : str, optional
            Filename to save plot to
        """
        try:
            # Get equity curve data
            equity = results['equity_curve']
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot equity curve
            plt.plot(equity, label='Equity Curve')
            
            # Add buy and sell points
            for timestamp, price in results['buy_points']:
                try:
                    index = equity.index(timestamp)
                    plt.scatter(index, price, color='green', marker='^', s=100)
                except (ValueError, TypeError):
                    pass
                    
            for timestamp, price in results['sell_points']:
                try:
                    index = equity.index(timestamp)
                    plt.scatter(index, price, color='red', marker='v', s=100)
                except (ValueError, TypeError):
                    pass
            
            # Add title and labels
            plt.title(f"{results['symbol']} - {results['strategy']} Strategy")
            plt.xlabel('Time')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.legend()
            
            # Save figure if filename is provided
            if filename:
                plt.savefig(filename)
                logger.info(f"Plot saved to {filename}")
            
            # Show figure
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")

def main():
    """Run a backtest from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--strategy', type=str, default='adaptive', help='Strategy name (moving_average_crossover, rsi_reversal, or adaptive)')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, default='5minute', help='Data timeframe')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--plot', type=str, help='Output file for plot')
    
    args = parser.parse_args()
    
    # Create backtester
    backtester = Backtester(initial_capital=args.capital)
    
    # Authenticate with Angel One API
    if not backtester.api.authenticate():
        logger.error("Failed to authenticate with Angel One API")
        return
    
    # Run backtest
    if args.strategy == 'adaptive':
        results = backtester.run_adaptive_strategy_backtest(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe
        )
    else:
        results = backtester.run_backtest(
            symbol=args.symbol,
            strategy_name=args.strategy,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe
        )
    
    # Check for errors
    if 'error' in results:
        logger.error(results['error'])
        return
    
    # Print results
    print(f"Symbol: {args.symbol}")
    print(f"Strategy: {args.strategy}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Total trades: {results['total_trades']}")
    print(f"Win rate: {results['win_rate']:.2%}")
    print(f"Net return: {results['net_return']:.2%}")
    print(f"Annualized return: {results.get('annualized_return', 0):.2%}")
    print(f"Max drawdown: {results['max_drawdown']:.2%}")
    
    # Save results if output file is provided
    if args.output:
        backtester.save_results(results, args.output)
    
    # Plot equity curve if plot file is provided
    if args.plot:
        backtester.plot_equity_curve(results, args.plot)

if __name__ == '__main__':
    main()