"""
Main Module
=======================================
Main entry point for the trading bot application.
"""

import os
import time
import schedule
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

from trading_bot import TradingBot
from utils.logger import setup_logger, trade_logger
from utils.telegram import send_status_notification, send_error_notification
import config

# Set up logger
logger = setup_logger('main')

def check_market_day() -> bool:
    """
    Check if today is a market day.
    
    Returns:
    --------
    bool
        True if today is a market day, False otherwise
    """
    # For simplicity, we'll assume the market is open Monday to Friday
    # A more accurate implementation would check for holidays
    weekday = datetime.now().weekday()
    return weekday < 5  # 0-4 are Monday to Friday

def run_trading_bot(bot: TradingBot) -> None:
    """
    Run the trading bot.
    
    Parameters:
    -----------
    bot : TradingBot
        Trading bot instance
    """
    if not check_market_day():
        logger.info("Not a market day, skipping")
        return
    
    logger.info("Running trading bot")
    
    try:
        # Run the trading bot
        bot.run_once()
    except Exception as e:
        logger.error(f"Error running trading bot: {str(e)}")
        send_error_notification(f"Error running trading bot: {str(e)}")

def close_all_positions(bot: TradingBot) -> None:
    """
    Close all positions.
    
    Parameters:
    -----------
    bot : TradingBot
        Trading bot instance
    """
    if not check_market_day():
        return
    
    logger.info("Closing all positions")
    
    try:
        # Close all positions
        bot.close_all_positions()
    except Exception as e:
        logger.error(f"Error closing positions: {str(e)}")
        send_error_notification(f"Error closing positions: {str(e)}")

def schedule_tasks(bot: TradingBot) -> None:
    """
    Schedule tasks for the trading bot.
    
    Parameters:
    -----------
    bot : TradingBot
        Trading bot instance
    """
    # Get market timing
    market_start = config.get('MARKET_START_TIME', '09:30:00')
    market_end = config.get('MARKET_END_TIME', '15:00:00')
    trade_exit = config.get('TRADE_EXIT_TIME', '15:00:00')
    
    # Schedule tasks
    # Run trading bot every 5 minutes during market hours
    schedule.every().day.at(market_start).do(run_trading_bot, bot=bot)
    schedule.every(5).minutes.do(run_trading_bot, bot=bot)
    
    # Close all positions at exit time
    schedule.every().day.at(trade_exit).do(close_all_positions, bot=bot)
    
    logger.info(f"Scheduled tasks: Start at {market_start}, check every 5 minutes, close all positions at {trade_exit}")

def main():
    """Main entry point for the trading bot application."""
    parser = argparse.ArgumentParser(description='Intraday Trading Bot for Indian Stocks')
    parser.add_argument('--watchlist', type=str, help='Path to watchlist file')
    parser.add_argument('--backtest', action='store_true', help='Run backtest instead of live trading')
    parser.add_argument('--symbol', type=str, help='Symbol for backtesting')
    parser.add_argument('--start-date', type=str, help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtesting (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Load watchlist
    watchlist = None
    if args.watchlist:
        watchlist = config.load_watchlist(args.watchlist)
    
    if args.backtest:
        # Run backtest
        from backtest import Backtester
        
        if not args.symbol:
            logger.error("Symbol is required for backtesting")
            return
        
        if not args.start_date:
            logger.error("Start date is required for backtesting")
            return
        
        if not args.end_date:
            # Default to today
            args.end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Running backtest for {args.symbol} from {args.start_date} to {args.end_date}")
        
        backtester = Backtester()
        
        # Authenticate with Angel One API
        if not backtester.api.authenticate():
            logger.error("Failed to authenticate with Angel One API")
            return
        
        # Run adaptive strategy backtest
        results = backtester.run_adaptive_strategy_backtest(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Check for errors
        if 'error' in results:
            logger.error(results['error'])
            return
        
        # Print results
        print(f"Symbol: {args.symbol}")
        print(f"Strategy: adaptive")
        print(f"Period: {args.start_date} to {args.end_date}")
        print(f"Total trades: {results['total_trades']}")
        print(f"Win rate: {results['win_rate']:.2%}")
        print(f"Net return: {results['net_return']:.2%}")
        print(f"Annualized return: {results.get('annualized_return', 0):.2%}")
        print(f"Max drawdown: {results['max_drawdown']:.2%}")
        
        # Save results
        output_file = f"data/backtest_{args.symbol}_{args.start_date}_{args.end_date}.json"
        backtester.save_results(results, output_file)
        
        # Plot equity curve
        plot_file = f"data/backtest_{args.symbol}_{args.start_date}_{args.end_date}.png"
        backtester.plot_equity_curve(results, plot_file)
        
    else:
        # Run live trading
        logger.info("Starting the trading bot application")
        
        # Send start notification
        send_status_notification("Trading bot application started")
        
        # Create trading bot
        bot = TradingBot(watchlist)
        
        # Schedule tasks
        schedule_tasks(bot)
        
        try:
            # Authenticate with Angel One API
            if not bot.authenticate():
                logger.error("Failed to authenticate with Angel One API")
                send_error_notification("Failed to authenticate with Angel One API")
                return
            
            logger.info("Authentication successful")
            
            # Run scheduled tasks
            while True:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Trading bot application stopped")
            send_status_notification("Trading bot application stopped")
            
        except Exception as e:
            logger.error(f"Critical error in trading bot application: {str(e)}")
            send_error_notification(f"Critical error in trading bot application: {str(e)}")

if __name__ == '__main__':
    main()