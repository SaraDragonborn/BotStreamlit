"""
Test script for backtesting functionality.
"""

import os
import sys
from datetime import datetime, timedelta
from backtest import Backtester
import config
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_backtest')

def test_strategy(symbol, strategy_name, start_date=None, end_date=None):
    """Test a strategy on a symbol."""
    # Default to last 30 days if dates not provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Testing {strategy_name} on {symbol} from {start_date} to {end_date}")
    
    # Create backtester
    backtester = Backtester()
    
    # Run backtest
    result = backtester.run_backtest(
        symbol=symbol,
        strategy_name=strategy_name,
        start_date=start_date,
        end_date=end_date
    )
    
    # Print results
    if 'error' in result:
        logger.error(result['error'])
        return
    
    print(f"\nResults for {symbol} with {strategy_name}:")
    print(f"Period: {start_date} to {end_date}")
    print(f"Total trades: {result['total_trades']}")
    print(f"Win rate: {result['win_rate']:.2%}")
    print(f"Net return: {result['net_return']:.2%}")
    print(f"Max drawdown: {result['max_drawdown']:.2%}")
    
    # Save results
    output_file = f"data/backtest_{symbol}_{strategy_name}_{start_date}_{end_date}.json"
    backtester.save_results(result, output_file)
    
    # Plot equity curve
    plot_file = f"data/backtest_{symbol}_{strategy_name}_{start_date}_{end_date}.png"
    backtester.plot_equity_curve(result, plot_file)
    
    return result

def test_adaptive_strategy(symbol, start_date=None, end_date=None):
    """Test the adaptive strategy on a symbol."""
    # Default to last 30 days if dates not provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Testing adaptive strategy on {symbol} from {start_date} to {end_date}")
    
    # Create backtester
    backtester = Backtester()
    
    # Run adaptive strategy backtest
    result = backtester.run_adaptive_strategy_backtest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Print results
    if 'error' in result:
        logger.error(result['error'])
        return
    
    print(f"\nResults for {symbol} with adaptive strategy:")
    print(f"Period: {start_date} to {end_date}")
    print(f"Total trades: {result['total_trades']}")
    print(f"Win rate: {result['win_rate']:.2%}")
    print(f"Net return: {result['net_return']:.2%}")
    print(f"Max drawdown: {result['max_drawdown']:.2%}")
    
    # Print strategy usage
    if 'strategy_usage' in result:
        print("\nStrategy usage:")
        for strategy, count in result['strategy_usage'].items():
            print(f"  {strategy}: {count} times")
    
    # Save results
    output_file = f"data/backtest_{symbol}_adaptive_{start_date}_{end_date}.json"
    backtester.save_results(result, output_file)
    
    # Plot equity curve
    plot_file = f"data/backtest_{symbol}_adaptive_{start_date}_{end_date}.png"
    backtester.plot_equity_curve(result, plot_file)
    
    return result

def test_strategy_comparison(symbol, start_date=None, end_date=None):
    """Compare all strategies on a symbol."""
    # Default to last 30 days if dates not provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Comparing strategies on {symbol} from {start_date} to {end_date}")
    
    # Create backtester
    backtester = Backtester()
    
    # Run strategy comparison
    result = backtester.run_strategy_comparison(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Print results
    if 'error' in result:
        logger.error(result['error'])
        return
    
    print(f"\nStrategy comparison for {symbol}:")
    print(f"Period: {start_date} to {end_date}")
    print(f"Best strategy: {result['best_strategy']}")
    
    for strategy, strategy_result in result.items():
        if strategy == 'best_strategy':
            continue
        
        print(f"\n{strategy}:")
        print(f"  Total trades: {strategy_result['total_trades']}")
        print(f"  Win rate: {strategy_result['win_rate']:.2%}")
        print(f"  Net return: {strategy_result['net_return']:.2%}")
        print(f"  Max drawdown: {strategy_result['max_drawdown']:.2%}")
    
    # Save results
    output_file = f"data/backtest_{symbol}_comparison_{start_date}_{end_date}.json"
    backtester.save_results(result, output_file)
    
    return result

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test backtesting functionality')
    parser.add_argument('--symbol', type=str, default='RELIANCE', help='Stock symbol')
    parser.add_argument('--strategy', type=str, default='adaptive', help='Strategy to test (moving_average_crossover, rsi_reversal, adaptive, or comparison)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Run backtest
    if args.strategy == 'moving_average_crossover':
        test_strategy(args.symbol, 'moving_average_crossover', args.start_date, args.end_date)
    elif args.strategy == 'rsi_reversal':
        test_strategy(args.symbol, 'rsi_reversal', args.start_date, args.end_date)
    elif args.strategy == 'adaptive':
        test_adaptive_strategy(args.symbol, args.start_date, args.end_date)
    elif args.strategy == 'comparison':
        test_strategy_comparison(args.symbol, args.start_date, args.end_date)
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
        sys.exit(1)

if __name__ == '__main__':
    # Check if API credentials are set
    if not config.get('ANGEL_ONE_API_KEY'):
        logger.error("API credentials not set. Please set them in .env file.")
        sys.exit(1)
    
    main()