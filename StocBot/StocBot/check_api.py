"""
Script to check API connection and get market data.
"""

import os
import sys
import json
from datetime import datetime, timedelta
import logging
from api.angel_api import AngelOneAPI
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('check_api')

def check_auth():
    """Check authentication with Angel One API."""
    api = AngelOneAPI()
    
    logger.info("Checking authentication...")
    if api.authenticate():
        logger.info("Authentication successful")
        return api
    else:
        logger.error("Authentication failed")
        return None

def check_profile(api):
    """Check user profile."""
    logger.info("Checking user profile...")
    profile = api.get_profile()
    
    if profile:
        logger.info("Profile retrieved successfully")
        print(f"\nProfile:")
        print(f"  Name: {profile.get('name', 'N/A')}")
        print(f"  Email: {profile.get('email', 'N/A')}")
        print(f"  Mobile: {profile.get('mobile', 'N/A')}")
        return True
    else:
        logger.error("Failed to retrieve profile")
        return False

def check_funds(api):
    """Check user funds."""
    logger.info("Checking user funds...")
    funds = api.get_funds()
    
    if funds:
        logger.info("Funds retrieved successfully")
        print(f"\nFunds:")
        print(f"  Available cash: {funds.get('availablecash', 'N/A')}")
        print(f"  Used margin: {funds.get('utilizedmargin', 'N/A')}")
        print(f"  Available margin: {funds.get('availablemargin', 'N/A')}")
        return True
    else:
        logger.error("Failed to retrieve funds")
        return False

def check_quote(api, symbol='RELIANCE'):
    """Check stock quote."""
    logger.info(f"Checking quote for {symbol}...")
    quote = api.get_quote(symbol)
    
    if quote:
        logger.info("Quote retrieved successfully")
        print(f"\nQuote for {symbol}:")
        print(f"  Last price: {quote.get('ltp', 'N/A')}")
        print(f"  Change: {quote.get('change', 'N/A')}")
        print(f"  Change percent: {quote.get('chgpcnt', 'N/A')}%")
        print(f"  Open: {quote.get('open', 'N/A')}")
        print(f"  High: {quote.get('high', 'N/A')}")
        print(f"  Low: {quote.get('low', 'N/A')}")
        print(f"  Close: {quote.get('close', 'N/A')}")
        print(f"  Volume: {quote.get('volume', 'N/A')}")
        return True
    else:
        logger.error(f"Failed to retrieve quote for {symbol}")
        return False

def check_historical_data(api, symbol='RELIANCE', timeframe='5minute'):
    """Check historical data."""
    logger.info(f"Checking historical data for {symbol}...")
    
    # Get data for the last 7 days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    data = api.get_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        from_date=start_date,
        to_date=end_date
    )
    
    if data is not None and not data.empty:
        logger.info("Historical data retrieved successfully")
        print(f"\nHistorical data for {symbol} ({timeframe}):")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Number of candles: {len(data)}")
        print(f"  First candle: {data.index[0]}")
        print(f"  Last candle: {data.index[-1]}")
        
        # Print latest candles
        print("\nLatest candles:")
        print(data.tail(5))
        
        return True
    else:
        logger.error(f"Failed to retrieve historical data for {symbol}")
        return False

def check_watchlist():
    """Check watchlist data."""
    logger.info("Checking watchlist...")
    
    # Load watchlist
    watchlist = config.load_watchlist('data/watchlist.txt')
    
    if watchlist:
        logger.info(f"Watchlist loaded successfully with {len(watchlist)} symbols")
        print(f"\nWatchlist:")
        for symbol in watchlist:
            print(f"  {symbol}")
        return True
    else:
        logger.error("Failed to load watchlist")
        return False

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check API connection and market data')
    parser.add_argument('--symbol', type=str, default='RELIANCE', help='Stock symbol to check')
    parser.add_argument('--timeframe', type=str, default='5minute', help='Timeframe for historical data')
    
    args = parser.parse_args()
    
    # Check if API credentials are set
    if not config.get('ANGEL_ONE_API_KEY'):
        logger.error("API credentials not set. Please set them in .env file.")
        sys.exit(1)
    
    # Run checks
    api = check_auth()
    if not api:
        sys.exit(1)
    
    check_profile(api)
    check_funds(api)
    check_quote(api, args.symbol)
    check_historical_data(api, args.symbol, args.timeframe)
    check_watchlist()
    
    print("\nAll checks completed")

if __name__ == '__main__':
    main()