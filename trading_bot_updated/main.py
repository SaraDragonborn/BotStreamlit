"""
Multi-Asset Trading Bot
=======================================
Main entry point for the trading bot.
"""

import os
import time
import argparse
import logging
import schedule
import datetime
import json
import pandas as pd
from typing import Dict, List, Optional, Union, Any

# Import utility modules
from utils.logger import get_trade_logger, setup_logger
from config import get_config, ASSET_CLASSES

# Import core components
from data_collector import data_collector
from strategy_selector import StrategySelector
from risk_manager import RiskManager
from trade_executor import trade_executor
from performance_tracker import performance_tracker

# Import API adapters
from api.alpaca_api import alpaca_api
from api.angel_api import angel_api

# Set up logging
logger = get_trade_logger()
main_logger = setup_logger('main', 'logs/main.log')

# Load configuration
config = get_config()

class TradingBot:
    """
    Multi-Asset Trading Bot.
    
    Manages the complete trading workflow:
    - Data collection
    - Strategy selection
    - Risk management
    - Trade execution
    - Performance tracking
    """
    
    def __init__(self, 
                paper_trading: bool = True,
                markets: List[str] = None,
                strategy_mode: str = 'adaptive'):
        """
        Initialize the Trading Bot.
        
        Parameters:
        -----------
        paper_trading : bool, default=True
            Whether to use paper trading (simulated) or live trading
        markets : List[str], optional
            Markets to trade ('US', 'India', 'Crypto', 'Forex')
        strategy_mode : str, default='adaptive'
            Strategy selection mode ('performance', 'rotation', 'ensemble', 'adaptive')
        """
        self.paper_trading = paper_trading
        self.markets = markets or ['US']  # Default to US market
        self.running = False
        self.last_run_time = None
        
        # Initialize strategy selector
        self.strategy_selector = StrategySelector(selection_mode=strategy_mode)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            max_portfolio_risk=config['trading']['max_portfolio_risk'],
            max_position_risk=config['trading']['max_position_risk'],
            target_positions=config['trading']['max_positions']
        )
        
        # Configure trade executor
        trade_executor.paper_trading = paper_trading
        
        main_logger.info(f"Trading Bot initialized with {len(self.markets)} markets, "
                       f"paper_trading={paper_trading}, strategy_mode={strategy_mode}")
    
    def start(self) -> None:
        """Start the trading bot."""
        if self.running:
            main_logger.warning("Trading Bot is already running")
            return
        
        self.running = True
        main_logger.info("Trading Bot started")
        
        try:
            # Run initial portfolio update
            self._update_portfolio_state()
            
            # Run initial market check
            self._check_markets()
            
            # Main loop
            while self.running:
                now = datetime.datetime.now()
                
                # Check if markets are open
                markets_open = self._check_markets()
                
                if any(markets_open.values()):
                    # Update portfolio state
                    self._update_portfolio_state()
                    
                    # Run trading cycle
                    self._trading_cycle()
                    
                    # Update last run time
                    self.last_run_time = now
                    
                    # Log status
                    open_markets = [market for market, is_open in markets_open.items() if is_open]
                    main_logger.info(f"Trading cycle complete. Active markets: {', '.join(open_markets)}")
                else:
                    main_logger.info("All markets closed, waiting...")
                
                # Sleep between cycles (5 minutes)
                time.sleep(300)
                
        except KeyboardInterrupt:
            main_logger.info("Trading Bot stopped by user")
        except Exception as e:
            main_logger.error(f"Error in Trading Bot main loop: {str(e)}")
        finally:
            self.running = False
            main_logger.info("Trading Bot stopped")
    
    def stop(self) -> None:
        """Stop the trading bot."""
        if not self.running:
            main_logger.warning("Trading Bot is not running")
            return
        
        self.running = False
        main_logger.info("Trading Bot stop signal received")
    
    def run_once(self) -> None:
        """Run a single trading cycle."""
        try:
            main_logger.info("Running single trading cycle")
            
            # Update portfolio state
            self._update_portfolio_state()
            
            # Check markets
            markets_open = self._check_markets()
            open_markets = [market for market, is_open in markets_open.items() if is_open]
            
            if not open_markets:
                main_logger.warning("No markets are open, running in simulation mode")
            
            # Run trading cycle
            self._trading_cycle()
            
            # Update last run time
            self.last_run_time = datetime.datetime.now()
            
            main_logger.info("Single trading cycle complete")
            
        except Exception as e:
            main_logger.error(f"Error in single trading cycle: {str(e)}")
    
    def _check_markets(self) -> Dict[str, bool]:
        """
        Check if markets are open.
        
        Returns:
        --------
        Dict[str, bool]
            Market open status
        """
        market_status = {}
        
        try:
            # Check US market
            if 'US' in self.markets:
                if self.paper_trading:
                    # For paper trading, simulate market hours
                    now = datetime.datetime.now()
                    is_weekday = now.weekday() < 5  # Monday to Friday
                    is_market_hours = 9 <= now.hour < 16  # 9:00 AM to 4:00 PM
                    
                    market_status['US'] = is_weekday and is_market_hours
                else:
                    # Check Alpaca clock
                    clock_result = alpaca_api.get_clock()
                    market_status['US'] = clock_result.get('success', False) and clock_result.get('clock', {}).get('is_open', False)
            
            # Check Indian market
            if 'India' in self.markets:
                # Using a simple time-based check (9:15 AM - 3:30 PM IST on weekdays)
                now = datetime.datetime.now()
                india_time = now + datetime.timedelta(hours=5, minutes=30)  # Crude IST conversion
                is_weekday = india_time.weekday() < 5  # Monday to Friday
                is_market_hours = (
                    (india_time.hour > 9 or (india_time.hour == 9 and india_time.minute >= 15)) and
                    (india_time.hour < 15 or (india_time.hour == 15 and india_time.minute <= 30))
                )
                
                market_status['India'] = is_weekday and is_market_hours
            
            # For Crypto and Forex, markets are always open
            if 'Crypto' in self.markets:
                market_status['Crypto'] = True
            
            if 'Forex' in self.markets:
                # Forex is open 24/5
                now = datetime.datetime.now()
                is_weekday = now.weekday() < 5  # Monday to Friday
                market_status['Forex'] = is_weekday
            
        except Exception as e:
            main_logger.error(f"Error checking market status: {str(e)}")
            # Default to closed for safety
            for market in self.markets:
                market_status[market] = False
        
        # Log market status
        status_str = ", ".join([f"{m}: {'Open' if status else 'Closed'}" for m, status in market_status.items()])
        main_logger.info(f"Market status: {status_str}")
        
        return market_status
    
    def _update_portfolio_state(self) -> None:
        """Update portfolio state across all markets."""
        try:
            portfolio_value = 0.0
            buying_power = 0.0
            positions = {}
            
            # Get US portfolio data
            if 'US' in self.markets:
                # Get account info
                account_result = alpaca_api.get_account()
                
                if account_result.get('success', False):
                    account_info = account_result.get('account_info', {})
                    portfolio_value += float(account_info.get('portfolio_value', 0))
                    buying_power += float(account_info.get('buying_power', 0))
                
                # Get positions
                positions_result = alpaca_api.get_positions()
                
                if positions_result.get('success', False):
                    for position in positions_result.get('positions', []):
                        symbol = position.get('symbol')
                        positions[symbol] = {
                            'market': 'US',
                            'symbol': symbol,
                            'quantity': float(position.get('qty', 0)),
                            'entry_price': float(position.get('avg_entry_price', 0)),
                            'current_price': float(position.get('current_price', 0)),
                            'market_value': float(position.get('market_value', 0)),
                            'unrealized_pl': float(position.get('unrealized_pl', 0)),
                            'unrealized_plpc': float(position.get('unrealized_plpc', 0))
                        }
            
            # Get Indian portfolio data
            if 'India' in self.markets:
                # Get funds
                funds_result = angel_api.get_funds()
                
                if funds_result.get('status', False):
                    funds_data = funds_result.get('data', {})
                    if not self.paper_trading:
                        # Only add to portfolio value if using live trading
                        portfolio_value += float(funds_data.get('availablecash', 0))
                        buying_power += float(funds_data.get('availablecash', 0))
                
                # Get positions
                positions_result = angel_api.get_positions()
                
                if positions_result.get('status', False):
                    for position in positions_result.get('data', []):
                        symbol = position.get('tradingsymbol')
                        positions[symbol] = {
                            'market': 'India',
                            'symbol': symbol,
                            'quantity': float(position.get('netqty', 0)),
                            'entry_price': float(position.get('averageprice', 0)),
                            'current_price': float(position.get('ltp', 0)),
                            'market_value': float(position.get('netqty', 0)) * float(position.get('ltp', 0)),
                            'unrealized_pl': float(position.get('mtm', 0)),
                            'unrealized_plpc': float(position.get('mtm', 0)) / (float(position.get('netqty', 0)) * float(position.get('averageprice', 0))) if float(position.get('netqty', 0)) * float(position.get('averageprice', 0)) > 0 else 0
                        }
            
            # Get sector data for risk analysis
            symbols = list(positions.keys())
            if symbols:
                try:
                    sector_data = data_collector.get_sector_data(symbols)
                except:
                    sector_data = {symbol: 'Unknown' for symbol in symbols}
            else:
                sector_data = {}
            
            # Update risk manager
            self.risk_manager.update_portfolio_state(
                portfolio_value=portfolio_value,
                buying_power=buying_power,
                positions=positions,
                sector_data=sector_data
            )
            
            # Update equity in performance tracker
            performance_tracker.update_equity(portfolio_value)
            
            main_logger.info(f"Portfolio updated: Value=${portfolio_value:.2f}, "
                           f"Buying power=${buying_power:.2f}, "
                           f"Positions={len(positions)}")
            
        except Exception as e:
            main_logger.error(f"Error updating portfolio state: {str(e)}")
    
    def _trading_cycle(self) -> None:
        """Run a complete trading cycle."""
        try:
            # Check if we should exit all positions due to risk
            if self.risk_manager.should_exit_all_positions():
                main_logger.warning("Risk limit exceeded, exiting all positions")
                self._exit_all_positions()
                return
            
            # Get tradable symbols for each market
            tradable_symbols = self._get_tradable_symbols()
            
            # Process each market
            for market, symbols in tradable_symbols.items():
                if not symbols:
                    continue
                
                main_logger.info(f"Processing {market} market with {len(symbols)} symbols")
                
                # Get market data
                market_data = data_collector.get_market_data(market)
                
                # Skip if market is closed (safety check)
                if not market_data.get('is_open', False) and not self.paper_trading:
                    main_logger.info(f"Market {market} is closed, skipping")
                    continue
                
                # Process each symbol
                for symbol in symbols:
                    try:
                        # Get data for this symbol
                        data = data_collector.get_stock_data_with_indicators(
                            symbol=symbol,
                            timeframe='1d',
                            limit=100,
                            source='auto',
                            indicators=[
                                {'name': 'sma', 'params': {'period': 20}},
                                {'name': 'sma', 'params': {'period': 50}},
                                {'name': 'sma', 'params': {'period': 200}},
                                {'name': 'rsi', 'params': {'period': 14}},
                                {'name': 'bollinger', 'params': {'period': 20, 'std_dev': 2.0}},
                                {'name': 'macd', 'params': {}}
                            ]
                        )
                        
                        if data.empty:
                            main_logger.warning(f"No data for {symbol}, skipping")
                            continue
                        
                        # Generate signals using strategy selector
                        signals = self.strategy_selector.generate_signals(data)
                        
                        # Check current position
                        position = None
                        if market == 'US':
                            position_result = alpaca_api.get_position(symbol)
                            if position_result.get('success', False):
                                position = position_result.get('position')
                        elif market == 'India':
                            positions_result = angel_api.get_positions()
                            if positions_result.get('status', False):
                                for pos in positions_result.get('data', []):
                                    if pos.get('tradingsymbol') == symbol:
                                        position = pos
                                        break
                        
                        # Process the latest signal
                        latest_signal = signals.iloc[-1]
                        signal_value = latest_signal.get('signal', 0)
                        
                        # Get current price
                        current_price = latest_signal.get('close')
                        
                        if signal_value == 1:  # Buy signal
                            # Only buy if we don't already have a position
                            if not position or float(position.get('qty', 0)) == 0:
                                # Calculate stop loss
                                stop_price = self.risk_manager.calculate_stop_loss(
                                    data=data,
                                    entry_price=current_price,
                                    direction='long'
                                )
                                
                                # Calculate position size
                                shares, position_info = self.risk_manager.calculate_position_size(
                                    symbol=symbol,
                                    entry_price=current_price,
                                    stop_price=stop_price
                                )
                                
                                if shares > 0:
                                    # Create signal dict
                                    signal_dict = {
                                        'symbol': symbol,
                                        'direction': 'buy',
                                        'signal_type': self.strategy_selector.selection_mode,
                                        'strength': 1.0
                                    }
                                    
                                    # Execute the trade
                                    trade_result = trade_executor.execute_signal(
                                        signal=signal_dict,
                                        risk_info=position_info,
                                        market=market
                                    )
                                    
                                    main_logger.info(f"Buy signal for {symbol}: {shares} shares at {current_price}")
                            else:
                                main_logger.info(f"Already have position in {symbol}, skipping buy signal")
                        
                        elif signal_value == -1:  # Sell signal
                            # Only sell if we have a position
                            if position and float(position.get('qty', 0)) > 0:
                                # Create signal dict
                                signal_dict = {
                                    'symbol': symbol,
                                    'direction': 'sell',
                                    'signal_type': self.strategy_selector.selection_mode,
                                    'strength': 1.0
                                }
                                
                                # Execute the trade
                                qty = float(position.get('qty', 0))
                                
                                risk_info = {
                                    'shares': qty,
                                    'entry_price': None  # Market order
                                }
                                
                                trade_result = trade_executor.execute_signal(
                                    signal=signal_dict,
                                    risk_info=risk_info,
                                    market=market
                                )
                                
                                # Record the trade in performance tracker
                                if trade_result.get('status') != 'failed':
                                    trade_record = {
                                        'symbol': symbol,
                                        'entry_date': None,  # We don't have this information
                                        'exit_date': datetime.datetime.now().isoformat(),
                                        'entry_price': float(position.get('avg_entry_price', 0)),
                                        'exit_price': current_price,
                                        'shares': qty,
                                        'direction': 'long',
                                        'pnl': qty * (current_price - float(position.get('avg_entry_price', 0))),
                                        'pnl_pct': (current_price / float(position.get('avg_entry_price', 0))) - 1 if float(position.get('avg_entry_price', 0)) > 0 else 0,
                                        'strategy': self.strategy_selector.selection_mode,
                                        'market': market
                                    }
                                    
                                    performance_tracker.add_trade(trade_record)
                                
                                main_logger.info(f"Sell signal for {symbol}: {qty} shares at {current_price}")
                            else:
                                main_logger.info(f"No position in {symbol}, skipping sell signal")
                    
                    except Exception as e:
                        main_logger.error(f"Error processing {symbol}: {str(e)}")
            
            # Update pending orders
            trade_executor.update_pending_orders()
            
            # Generate performance report
            self._generate_performance_report()
            
        except Exception as e:
            main_logger.error(f"Error in trading cycle: {str(e)}")
    
    def _get_tradable_symbols(self) -> Dict[str, List[str]]:
        """
        Get tradable symbols for each market.
        
        Returns:
        --------
        Dict[str, List[str]]
            Tradable symbols by market
        """
        tradable_symbols = {}
        
        # US symbols
        if 'US' in self.markets:
            tradable_symbols['US'] = config['us_stocks']
        
        # Indian symbols
        if 'India' in self.markets:
            tradable_symbols['India'] = config['indian_stocks']
        
        # Crypto symbols
        if 'Crypto' in self.markets:
            tradable_symbols['Crypto'] = config['crypto']
        
        # Forex symbols
        if 'Forex' in self.markets:
            tradable_symbols['Forex'] = config['forex']
        
        return tradable_symbols
    
    def _exit_all_positions(self) -> None:
        """Exit all positions across all markets."""
        try:
            # Exit US positions
            if 'US' in self.markets:
                positions_result = alpaca_api.get_positions()
                
                if positions_result.get('success', False):
                    for position in positions_result.get('positions', []):
                        symbol = position.get('symbol')
                        qty = float(position.get('qty', 0))
                        
                        if qty > 0:
                            # Create sell signal
                            signal_dict = {
                                'symbol': symbol,
                                'direction': 'sell',
                                'signal_type': 'risk_management',
                                'strength': 1.0
                            }
                            
                            risk_info = {
                                'shares': qty,
                                'entry_price': None  # Market order
                            }
                            
                            trade_result = trade_executor.execute_signal(
                                signal=signal_dict,
                                risk_info=risk_info,
                                market='US'
                            )
                            
                            main_logger.info(f"Emergency exit for {symbol}: {qty} shares")
            
            # Exit Indian positions
            if 'India' in self.markets:
                positions_result = angel_api.get_positions()
                
                if positions_result.get('status', False):
                    for position in positions_result.get('data', []):
                        symbol = position.get('tradingsymbol')
                        qty = float(position.get('netqty', 0))
                        
                        if qty > 0:
                            # Create sell signal
                            signal_dict = {
                                'symbol': symbol,
                                'direction': 'sell',
                                'signal_type': 'risk_management',
                                'strength': 1.0
                            }
                            
                            risk_info = {
                                'shares': qty,
                                'entry_price': None  # Market order
                            }
                            
                            trade_result = trade_executor.execute_signal(
                                signal=signal_dict,
                                risk_info=risk_info,
                                market='India'
                            )
                            
                            main_logger.info(f"Emergency exit for {symbol}: {qty} shares")
            
            main_logger.warning("All positions exited")
            
        except Exception as e:
            main_logger.error(f"Error exiting all positions: {str(e)}")
    
    def _generate_performance_report(self) -> None:
        """Generate and save performance report."""
        try:
            # Only generate report occasionally (once per day)
            if self.last_run_time:
                today = datetime.datetime.now().date()
                last_run_date = self.last_run_time.date()
                
                if today == last_run_date:
                    return
            
            # Generate report
            report_file = f"data/performance/report_{datetime.datetime.now().strftime('%Y-%m-%d')}.json"
            performance_tracker.export_performance_report(report_file)
            
            # Generate plots
            plots_dir = "data/performance/plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            equity_plot = f"{plots_dir}/equity_{datetime.datetime.now().strftime('%Y-%m-%d')}.png"
            pnl_plot = f"{plots_dir}/pnl_{datetime.datetime.now().strftime('%Y-%m-%d')}.png"
            strategy_plot = f"{plots_dir}/strategy_{datetime.datetime.now().strftime('%Y-%m-%d')}.png"
            
            performance_tracker.plot_equity_curve(equity_plot)
            performance_tracker.plot_pnl_distribution(pnl_plot)
            performance_tracker.plot_strategy_comparison(strategy_plot)
            
            main_logger.info("Performance report and plots generated")
            
        except Exception as e:
            main_logger.error(f"Error generating performance report: {str(e)}")


def main():
    """Run the trading bot from command line."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Asset Trading Bot')
    
    parser.add_argument('--paper', action='store_true', help='Use paper trading (default)')
    parser.add_argument('--live', action='store_true', help='Use live trading (real money)')
    parser.add_argument('--markets', type=str, default='US',
                        help='Markets to trade (comma-separated: US,India,Crypto,Forex)')
    parser.add_argument('--strategy', type=str, default='adaptive',
                        help='Strategy selection mode (performance, rotation, ensemble, adaptive)')
    parser.add_argument('--run-once', action='store_true', help='Run a single trading cycle')
    parser.add_argument('--schedule', action='store_true', help='Run at scheduled times')
    
    args = parser.parse_args()
    
    # Force paper trading unless explicitly using live
    paper_trading = not args.live
    
    # Parse markets
    markets = args.markets.split(',')
    
    # Initialize trading bot
    bot = TradingBot(
        paper_trading=paper_trading,
        markets=markets,
        strategy_mode=args.strategy
    )
    
    # Display warning for live trading
    if not paper_trading:
        print("="*80)
        print("WARNING: LIVE TRADING ENABLED. REAL FUNDS WILL BE USED!")
        print("="*80)
        
        # Confirm
        confirm = input("Are you sure you want to continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Live trading canceled.")
            return
    
    # Run based on arguments
    if args.run_once:
        print("Running single trading cycle...")
        bot.run_once()
        print("Done.")
    elif args.schedule:
        print(f"Scheduling trading bot for {'paper' if paper_trading else 'LIVE'} trading...")
        
        # Schedule for US market
        if 'US' in markets:
            # Pre-market check (9:00 AM EST)
            schedule.every().day.at("09:00").do(bot.run_once)
            # Market open (9:30 AM EST)
            schedule.every().day.at("09:30").do(bot.run_once)
            # Mid-day check (12:00 PM EST)
            schedule.every().day.at("12:00").do(bot.run_once)
            # Market close approach (3:45 PM EST)
            schedule.every().day.at("15:45").do(bot.run_once)
        
        # Schedule for Indian market
        if 'India' in markets:
            # Convert to local time (example assumes EST)
            # Market open (9:15 AM IST ≈ 11:45 PM EST previous day)
            schedule.every().day.at("23:45").do(bot.run_once)
            # Mid-day check (12:00 PM IST ≈ 2:30 AM EST)
            schedule.every().day.at("02:30").do(bot.run_once)
            # Market close approach (3:15 PM IST ≈ 5:45 AM EST)
            schedule.every().day.at("05:45").do(bot.run_once)
        
        print("Bot scheduled. Press Ctrl+C to exit.")
        
        # Run the schedule
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("Scheduler stopped.")
    else:
        # Run continuously
        print(f"Starting trading bot for {'paper' if paper_trading else 'LIVE'} trading...")
        print("Press Ctrl+C to stop.")
        bot.start()


if __name__ == '__main__':
    main()