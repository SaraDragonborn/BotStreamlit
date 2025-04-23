"""
Trading Bot Module
=======================================
Main bot class for coordinating market analysis, strategy selection, and trade execution.
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import config
from utils.logger import setup_logger, trade_logger
from utils.telegram import send_trade_notification, send_status_notification, send_error_notification
from api.angel_api import AngelOneAPI
from strategies import MovingAverageCrossover, RSIReversal, MarketConditionAnalyzer

# Set up logger
logger = setup_logger('trading_bot')

class TradingBot:
    """
    Trading Bot for Indian intraday stock trading.
    
    Coordinates market analysis, strategy selection, and trade execution.
    
    Attributes:
    -----------
    api : AngelOneAPI
        API connector for Angel One
    watchlist : list
        List of stocks to trade
    positions : dict
        Dictionary of current positions
    market_analyzer : MarketConditionAnalyzer
        Market condition analyzer
    """
    
    def __init__(self, watchlist: Optional[List[str]] = None):
        """
        Initialize the Trading Bot.
        
        Parameters:
        -----------
        watchlist : list, optional
            List of stocks to trade
        """
        self.api = AngelOneAPI()
        self.watchlist = watchlist or config.load_watchlist()
        self.positions = {}
        self.market_analyzer = MarketConditionAnalyzer()
        
        # Strategy instances
        self.strategies = {
            'moving_average_crossover': MovingAverageCrossover(),
            'rsi_reversal': RSIReversal()
        }
        
        # Trading parameters
        self.capital = config.get('CAPITAL', 100000)
        self.capital_per_trade = config.get('CAPITAL_PER_TRADE', 5000)
        self.max_positions = config.get('MAX_POSITIONS', 3)
        self.stop_loss_percent = config.get('STOP_LOSS_PERCENT', 1.5)
        self.target_percent = config.get('TARGET_PERCENT', 3.0)
        self.trailing_stop_threshold = config.get('TRAILING_STOP_THRESHOLD', 1.5)
        
        # Market timing
        self.market_start_time = config.get('MARKET_START_TIME', '09:30:00')
        self.market_end_time = config.get('MARKET_END_TIME', '15:00:00')
        self.trade_exit_time = config.get('TRADE_EXIT_TIME', '15:00:00')
        
        logger.info(f"Trading Bot initialized with {len(self.watchlist)} stocks in watchlist")
    
    def authenticate(self) -> bool:
        """
        Authenticate with Angel One API.
        
        Returns:
        --------
        bool
            True if authentication was successful, False otherwise
        """
        return self.api.authenticate()
    
    def check_market_hours(self) -> bool:
        """
        Check if it's market hours.
        
        Returns:
        --------
        bool
            True if it's market hours, False otherwise
        """
        now = datetime.now().time()
        start_time = datetime.strptime(self.market_start_time, '%H:%M:%S').time()
        end_time = datetime.strptime(self.market_end_time, '%H:%M:%S').time()
        
        return start_time <= now <= end_time
    
    def check_exit_time(self) -> bool:
        """
        Check if it's time to exit all positions.
        
        Returns:
        --------
        bool
            True if it's time to exit, False otherwise
        """
        now = datetime.now().time()
        exit_time = datetime.strptime(self.trade_exit_time, '%H:%M:%S').time()
        
        return now >= exit_time
    
    def filter_liquid_stocks(self) -> List[str]:
        """
        Filter watchlist for liquid stocks.
        
        Returns:
        --------
        list
            List of liquid stocks
        """
        liquid_stocks = []
        
        for symbol in self.watchlist:
            try:
                # Get latest quote
                quote = self.api.get_quote(symbol)
                
                if quote and 'volume' in quote:
                    volume = quote['volume']
                    
                    # Add stocks with good volume (adjust threshold as needed)
                    if volume > 100000:
                        liquid_stocks.append(symbol)
                
            except Exception as e:
                logger.error(f"Error checking liquidity for {symbol}: {str(e)}")
        
        return liquid_stocks
    
    def get_market_condition(self) -> str:
        """
        Get current market condition using the index.
        
        Returns:
        --------
        str
            Market condition ('trending', 'sideways', or 'neutral')
        """
        index_symbol = self.market_analyzer.params['index_symbol']
        
        try:
            # Get index data
            index_data = self.api.get_historical_data(
                symbol=index_symbol,
                timeframe='5minute',
                from_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                exchange='NSE'
            )
            
            if index_data is not None:
                return self.market_analyzer.analyze(index_data)
            
        except Exception as e:
            logger.error(f"Error getting market condition: {str(e)}")
        
        return 'neutral'
    
    def select_strategy(self, symbol: str) -> str:
        """
        Select the appropriate strategy based on market condition.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        str
            Strategy name ('moving_average_crossover' or 'rsi_reversal')
        """
        try:
            # Get historical data
            data = self.api.get_historical_data(
                symbol=symbol,
                timeframe='5minute',
                from_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                exchange='NSE'
            )
            
            if data is not None:
                # Get market condition and recommend strategy
                return self.market_analyzer.recommend_strategy(data)
            
        except Exception as e:
            logger.error(f"Error selecting strategy for {symbol}: {str(e)}")
        
        # Default to moving average crossover if error
        return 'moving_average_crossover'
    
    def get_signal(self, symbol: str, strategy_name: str) -> Optional[Dict]:
        """
        Get trading signal for a symbol using the specified strategy.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        strategy_name : str
            Strategy name
            
        Returns:
        --------
        dict or None
            Signal details or None if no signal
        """
        try:
            # Get historical data
            data = self.api.get_historical_data(
                symbol=symbol,
                timeframe='5minute',
                from_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                exchange='NSE'
            )
            
            if data is None:
                return None
            
            # Get strategy
            strategy = self.strategies.get(strategy_name)
            if strategy is None:
                logger.error(f"Strategy {strategy_name} not found")
                return None
            
            # Get signal
            signal = strategy.get_latest_signal(data)
            
            if signal in ['BUY', 'SELL']:
                # Get current price
                quote = self.api.get_quote(symbol)
                if quote is None:
                    return None
                
                price = quote.get('ltp', 0)
                
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'strategy': strategy_name,
                    'price': price,
                    'timestamp': datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting signal for {symbol}: {str(e)}")
            return None
    
    def calculate_position_size(self, symbol: str, price: float) -> int:
        """
        Calculate position size based on capital per trade.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        price : float
            Current price
            
        Returns:
        --------
        int
            Number of shares to trade
        """
        # Calculate based on capital per trade
        position_size = int(self.capital_per_trade / price)
        
        # Round to the nearest lot size
        # This depends on the symbol's lot size, which we don't have
        # For now, let's just return the calculated position size
        
        return max(1, position_size)  # Ensure at least 1 share
    
    def execute_trade(self, signal: Dict) -> bool:
        """
        Execute a trade based on the signal.
        
        Parameters:
        -----------
        signal : dict
            Signal details
            
        Returns:
        --------
        bool
            True if trade was executed successfully, False otherwise
        """
        symbol = signal['symbol']
        side = signal['signal']
        price = signal['price']
        strategy = signal['strategy']
        
        # Check if we already have a position for this symbol
        if symbol in self.positions:
            logger.info(f"Already have a position for {symbol}, skipping")
            return False
        
        # Check if we've reached the maximum number of positions
        if len(self.positions) >= self.max_positions and side == 'BUY':
            logger.info(f"Maximum positions ({self.max_positions}) reached, skipping buy for {symbol}")
            return False
        
        # Calculate position size
        quantity = self.calculate_position_size(symbol, price)
        
        try:
            # Execute the trade
            order_id = self.api.place_order(
                symbol=symbol,
                quantity=quantity,
                side=side,
                order_type='MARKET',
                stop_loss=self.stop_loss_percent if side == 'BUY' else None,
                target=self.target_percent if side == 'BUY' else None
            )
            
            if order_id:
                # Add to positions if it's a buy
                if side == 'BUY':
                    self.positions[symbol] = {
                        'order_id': order_id,
                        'quantity': quantity,
                        'entry_price': price,
                        'side': side,
                        'strategy': strategy,
                        'entry_time': datetime.now().isoformat()
                    }
                
                # Log the trade
                trade_data = {
                    'symbol': symbol,
                    'order_type': side,
                    'quantity': quantity,
                    'price': price,
                    'strategy': strategy,
                    'order_id': order_id
                }
                trade_logger.log_trade(trade_data)
                
                # Send notification
                send_trade_notification(side, symbol, quantity, price, strategy)
                
                logger.info(f"Trade executed: {side} {quantity} {symbol} @ {price}")
                return True
            else:
                logger.error(f"Failed to execute trade: {side} {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return False
    
    def update_positions(self) -> None:
        """
        Update current positions.
        """
        if not self.positions:
            return
        
        try:
            # Get current positions from API
            api_positions = self.api.get_positions() or []
            
            # Update positions
            for symbol, position in list(self.positions.items()):
                # Check if position is still open
                position_found = False
                
                for api_position in api_positions:
                    if api_position['tradingsymbol'] == symbol:
                        position_found = True
                        # Update position details
                        quantity = int(api_position['netqty'])
                        
                        if quantity == 0:
                            # Position is closed
                            logger.info(f"Position for {symbol} is closed")
                            del self.positions[symbol]
                        else:
                            # Update position
                            self.positions[symbol]['quantity'] = quantity
                
                if not position_found:
                    # Position not found, assume it's closed
                    logger.info(f"Position for {symbol} not found, assuming closed")
                    del self.positions[symbol]
                    
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
    
    def close_all_positions(self) -> bool:
        """
        Close all open positions.
        
        Returns:
        --------
        bool
            True if all positions were closed successfully, False otherwise
        """
        if not self.positions:
            logger.info("No positions to close")
            return True
        
        success = True
        
        # Close each position
        for symbol in list(self.positions.keys()):
            if not self.close_position(symbol):
                success = False
        
        return success
    
    def close_position(self, symbol: str) -> bool:
        """
        Close a specific position.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        bool
            True if position was closed successfully, False otherwise
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return False
        
        try:
            # Square off the position
            result = self.api.square_off_position(symbol)
            
            if result:
                # Remove from positions
                position = self.positions.pop(symbol)
                
                # Log the trade
                side = 'SELL' if position['side'] == 'BUY' else 'BUY'
                quantity = position['quantity']
                
                # Get current price
                quote = self.api.get_quote(symbol)
                price = quote['ltp'] if quote else 0
                
                trade_data = {
                    'symbol': symbol,
                    'order_type': side,
                    'quantity': quantity,
                    'price': price,
                    'strategy': position['strategy'],
                    'is_exit': True
                }
                trade_logger.log_trade(trade_data)
                
                # Send notification
                send_trade_notification(side, symbol, quantity, price, position['strategy'])
                
                logger.info(f"Position closed: {side} {quantity} {symbol} @ {price}")
                return True
            else:
                logger.error(f"Failed to close position for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")
            return False
    
    def run_once(self) -> None:
        """
        Run one iteration of the trading bot.
        """
        # Check if it's market hours
        if not self.check_market_hours():
            logger.info("Not market hours, skipping")
            return
        
        # Check if it's time to exit all positions
        if self.check_exit_time():
            logger.info("Exit time reached, closing all positions")
            self.close_all_positions()
            return
        
        # Authenticate with Angel One API if not already
        if not self.api._authenticated:
            if not self.authenticate():
                logger.error("Failed to authenticate with Angel One API")
                return
        
        # Update positions
        self.update_positions()
        
        # Filter for liquid stocks
        liquid_stocks = self.filter_liquid_stocks()
        logger.info(f"Found {len(liquid_stocks)} liquid stocks")
        
        # Get market condition
        market_condition = self.get_market_condition()
        logger.info(f"Current market condition: {market_condition}")
        
        # Process each stock
        for symbol in liquid_stocks:
            # Skip if we already have the maximum number of positions
            if len(self.positions) >= self.max_positions:
                logger.info(f"Maximum positions ({self.max_positions}) reached, skipping")
                break
            
            # Select strategy based on market condition
            if market_condition == 'trending':
                strategy_name = 'moving_average_crossover'
            elif market_condition == 'sideways':
                strategy_name = 'rsi_reversal'
            else:
                # Use symbol-specific strategy selection
                strategy_name = self.select_strategy(symbol)
            
            logger.info(f"Selected strategy for {symbol}: {strategy_name}")
            
            # Get signal
            signal = self.get_signal(symbol, strategy_name)
            
            if signal:
                logger.info(f"Signal for {symbol}: {signal['signal']}")
                
                # Execute trade
                self.execute_trade(signal)
            else:
                logger.info(f"No signal for {symbol}")
    
    def run(self) -> None:
        """
        Run the trading bot continuously.
        """
        logger.info("Starting trading bot")
        send_status_notification("Trading bot started")
        
        try:
            # Authenticate with Angel One API
            if not self.authenticate():
                logger.error("Failed to authenticate with Angel One API")
                send_error_notification("Failed to authenticate with Angel One API")
                return
            
            while True:
                try:
                    # Run one iteration
                    self.run_once()
                    
                    # Wait for 5 minutes
                    time.sleep(300)
                    
                except Exception as e:
                    logger.error(f"Error in trading bot iteration: {str(e)}")
                    send_error_notification(f"Error in trading bot iteration: {str(e)}")
                    
                    # Wait for 1 minute before retrying
                    time.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("Trading bot stopped")
            send_status_notification("Trading bot stopped")
            # Close all positions before exiting
            self.close_all_positions()
        
        except Exception as e:
            logger.error(f"Critical error in trading bot: {str(e)}")
            send_error_notification(f"Critical error in trading bot: {str(e)}")
            # Close all positions before exiting
            self.close_all_positions()