"""
Test script for trading bot functionality (paper trading mode).
"""

import os
import sys
import time
from datetime import datetime
import logging
from trading_bot import TradingBot
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_bot')

class PaperTradingBot(TradingBot):
    """
    Paper trading bot for testing without executing real trades.
    
    Extends the TradingBot class to override methods that would execute real trades.
    """
    
    def __init__(self, watchlist=None):
        """Initialize the paper trading bot."""
        super().__init__(watchlist)
        self.paper_positions = {}
        self.paper_orders = []
        logger.info("Paper trading mode enabled")
    
    def execute_trade(self, signal):
        """Override execute_trade to simulate trades instead of executing them."""
        symbol = signal['symbol']
        side = signal['signal']
        price = signal['price']
        strategy = signal['strategy']
        
        # Check if we already have a position for this symbol
        if symbol in self.paper_positions:
            logger.info(f"Already have a position for {symbol}, skipping")
            return False
        
        # Check if we've reached the maximum number of positions
        if len(self.paper_positions) >= self.max_positions and side == 'BUY':
            logger.info(f"Maximum positions ({self.max_positions}) reached, skipping buy for {symbol}")
            return False
        
        # Calculate position size
        quantity = self.calculate_position_size(symbol, price)
        
        # Simulate order execution
        order_id = f"paper-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.paper_orders) + 1}"
        
        # Add to paper orders
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'strategy': strategy,
            'status': 'COMPLETE',
            'timestamp': datetime.now().isoformat()
        }
        self.paper_orders.append(order)
        
        # Add to paper positions if it's a buy
        if side == 'BUY':
            self.paper_positions[symbol] = {
                'order_id': order_id,
                'quantity': quantity,
                'entry_price': price,
                'side': side,
                'strategy': strategy,
                'entry_time': datetime.now().isoformat()
            }
        
        # Log the trade
        logger.info(f"Paper trade executed: {side} {quantity} {symbol} @ {price}")
        
        return True
    
    def update_positions(self):
        """Override update_positions to update paper positions."""
        # In paper trading, we don't need to update positions from the API
        # But we could simulate price changes here if needed
        pass
    
    def close_position(self, symbol):
        """Override close_position to simulate closing a position."""
        if symbol not in self.paper_positions:
            logger.warning(f"No position found for {symbol}")
            return False
        
        try:
            # Get position details
            position = self.paper_positions.pop(symbol)
            
            # Get current price
            quote = self.api.get_quote(symbol)
            price = quote['ltp'] if quote else position['entry_price']
            
            # Calculate profit/loss
            side = 'SELL' if position['side'] == 'BUY' else 'BUY'
            quantity = position['quantity']
            entry_price = position['entry_price']
            exit_price = price
            pnl = (exit_price - entry_price) * quantity if side == 'SELL' else (entry_price - exit_price) * quantity
            
            # Create order for the close
            order_id = f"paper-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.paper_orders) + 1}"
            
            # Add to paper orders
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'strategy': position['strategy'],
                'status': 'COMPLETE',
                'timestamp': datetime.now().isoformat(),
                'is_exit': True,
                'entry_price': entry_price,
                'pnl': pnl,
                'pnl_percent': (pnl / (entry_price * quantity)) * 100
            }
            self.paper_orders.append(order)
            
            # Log the trade
            logger.info(f"Paper position closed: {side} {quantity} {symbol} @ {price}")
            logger.info(f"P&L: {pnl:.2f} ({order['pnl_percent']:.2%})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing paper position for {symbol}: {str(e)}")
            return False
    
    def close_all_positions(self):
        """Override close_all_positions to simulate closing all positions."""
        if not self.paper_positions:
            logger.info("No positions to close")
            return True
        
        success = True
        
        # Close each position
        for symbol in list(self.paper_positions.keys()):
            if not self.close_position(symbol):
                success = False
        
        return success
    
    def print_summary(self):
        """Print a summary of paper trading activity."""
        print("\n--- Paper Trading Summary ---")
        
        # Print current positions
        print("\nCurrent Positions:")
        if not self.paper_positions:
            print("  None")
        else:
            for symbol, position in self.paper_positions.items():
                print(f"  {symbol}: {position['quantity']} shares @ {position['entry_price']:.2f} ({position['strategy']})")
        
        # Print closed positions
        closed_orders = [order for order in self.paper_orders if order.get('is_exit', False)]
        print("\nClosed Positions:")
        if not closed_orders:
            print("  None")
        else:
            total_pnl = 0
            for order in closed_orders:
                print(f"  {order['symbol']}: {order['pnl']:.2f} ({order['pnl_percent']:.2%})")
                total_pnl += order['pnl']
            print(f"\nTotal P&L: {total_pnl:.2f}")
        
        # Print total trades
        print(f"\nTotal Trades: {len(self.paper_orders)}")
        
        # Print win/loss ratio
        if closed_orders:
            winning_trades = [order for order in closed_orders if order['pnl'] > 0]
            win_ratio = len(winning_trades) / len(closed_orders)
            print(f"Win Ratio: {win_ratio:.2%} ({len(winning_trades)}/{len(closed_orders)})")

def test_bot(watchlist_file=None, duration_minutes=30, interval_seconds=300):
    """Test the trading bot for a specified duration."""
    # Load watchlist
    watchlist = None
    if watchlist_file:
        watchlist = config.load_watchlist(watchlist_file)
    
    # Create paper trading bot
    bot = PaperTradingBot(watchlist)
    
    # Authenticate with Angel One API
    if not bot.api.authenticate():
        logger.error("Failed to authenticate with Angel One API")
        return
    
    logger.info("Authentication successful")
    
    # Calculate number of iterations
    iterations = int(duration_minutes * 60 / interval_seconds)
    
    # Run for specified duration
    logger.info(f"Running paper trading bot for {duration_minutes} minutes ({iterations} iterations)")
    
    try:
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            # Run one iteration
            bot.run_once()
            
            # Sleep for interval
            if i < iterations - 1:
                logger.info(f"Sleeping for {interval_seconds} seconds")
                time.sleep(interval_seconds)
        
        # Close all positions at the end
        logger.info("Test complete, closing all positions")
        bot.close_all_positions()
        
        # Print summary
        bot.print_summary()
        
    except KeyboardInterrupt:
        logger.info("Test interrupted")
        # Close all positions
        bot.close_all_positions()
        # Print summary
        bot.print_summary()
    
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        # Close all positions
        bot.close_all_positions()
        # Print summary
        bot.print_summary()

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trading bot functionality')
    parser.add_argument('--watchlist', type=str, default='data/watchlist.txt', help='Path to watchlist file')
    parser.add_argument('--duration', type=int, default=30, help='Test duration in minutes')
    parser.add_argument('--interval', type=int, default=300, help='Interval between iterations in seconds')
    
    args = parser.parse_args()
    
    test_bot(args.watchlist, args.duration, args.interval)

if __name__ == '__main__':
    # Check if API credentials are set
    if not config.get('ANGEL_ONE_API_KEY'):
        logger.error("API credentials not set. Please set them in .env file.")
        sys.exit(1)
    
    main()