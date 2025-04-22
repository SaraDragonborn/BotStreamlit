"""
Direct connection to Alpaca API without requiring backend server
"""

import requests
import pandas as pd
import os
import json
import time
from datetime import datetime, timedelta

# Alpaca API endpoints
ALPACA_ENDPOINTS = {
    'paper': 'https://paper-api.alpaca.markets',
    'live': 'https://api.alpaca.markets'
}

def test_alpaca_connection(api_key, api_secret, mode='paper'):
    """Test connection to Alpaca API directly"""
    try:
        if not api_key or not api_secret:
            return False, "API key and secret are required"
        
        base_url = ALPACA_ENDPOINTS.get(mode, ALPACA_ENDPOINTS['paper'])
        endpoint = f"{base_url}/v2/account"
        
        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        }
        
        response = requests.get(endpoint, headers=headers)
        
        if response.status_code == 200:
            return True, "Connected successfully to Alpaca API"
        else:
            return False, f"Connection failed: {response.text}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def get_account_info(api_key, api_secret, paper=True):
    """Get Alpaca account information directly"""
    try:
        mode = 'paper' if paper else 'live'
        base_url = ALPACA_ENDPOINTS.get(mode, ALPACA_ENDPOINTS['paper'])
        endpoint = f"{base_url}/v2/account"
        
        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        }
        
        response = requests.get(endpoint, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            result = {
                'portfolio_value': float(data.get('portfolio_value', 0)),
                'cash': float(data.get('cash', 0)),
                'buying_power': float(data.get('buying_power', 0)),
                'last_equity': float(data.get('last_equity', 0)),
                'pnl_percentage': 0  # Will calculate below
            }
            
            # Calculate P&L percentage
            if float(data.get('last_equity', 0)) > 0:
                equity_change = float(data.get('equity', 0)) - float(data.get('last_equity', 0))
                result['pnl_percentage'] = (equity_change / float(data.get('last_equity', 0))) * 100
            
            return result
        else:
            return {
                'portfolio_value': 0,
                'cash': 0,
                'buying_power': 0,
                'last_equity': 0,
                'pnl_percentage': 0
            }
    except Exception as e:
        print(f"Error fetching account info: {str(e)}")
        return {
            'portfolio_value': 0,
            'cash': 0,
            'buying_power': 0,
            'last_equity': 0,
            'pnl_percentage': 0
        }

def get_portfolio(api_key, api_secret, paper=True):
    """Get current positions directly from Alpaca"""
    try:
        mode = 'paper' if paper else 'live'
        base_url = ALPACA_ENDPOINTS.get(mode, ALPACA_ENDPOINTS['paper'])
        endpoint = f"{base_url}/v2/positions"
        
        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        }
        
        response = requests.get(endpoint, headers=headers)
        
        if response.status_code == 200:
            positions = response.json()
            result = []
            
            for position in positions:
                # Format each position
                formatted_position = {
                    'symbol': position.get('symbol', ''),
                    'qty': float(position.get('qty', 0)),
                    'market_value': float(position.get('market_value', 0)),
                    'unrealized_pl': float(position.get('unrealized_pl', 0)),
                    'unrealized_plpc': float(position.get('unrealized_plpc', 0)) * 100,  # Convert to percentage
                    'avg_entry_price': float(position.get('avg_entry_price', 0)),
                    'current_price': float(position.get('current_price', 0))
                }
                result.append(formatted_position)
            
            return result
        else:
            return []
    except Exception as e:
        print(f"Error fetching positions: {str(e)}")
        return []

def get_market_data(api_key, api_secret, paper=True):
    """Get market data for major indices via Alpaca API"""
    try:
        # Major indices ETFs
        indices = {
            'S&P 500': 'SPY',
            'Dow Jones': 'DIA',
            'Nasdaq': 'QQQ',
            'Russell 2000': 'IWM'
        }
        
        market_indices = {}
        
        for index_name, symbol in indices.items():
            # Get latest quote for each index ETF
            latest_data = get_latest_quote(api_key, api_secret, symbol, paper)
            
            if latest_data:
                # Get day percent change
                day_data = get_bars(api_key, api_secret, symbol, '1D', 2, paper)
                
                change_percent = 0
                if len(day_data) >= 2:
                    prev_close = day_data.iloc[-2]['close']
                    current_price = latest_data['price']
                    change_percent = ((current_price - prev_close) / prev_close) * 100
                
                market_indices[index_name] = {
                    'price': latest_data['price'],
                    'change_percent': change_percent
                }
        
        return market_indices
    except Exception as e:
        print(f"Error fetching market data: {str(e)}")
        return {}

def get_latest_quote(api_key, api_secret, symbol, paper=True):
    """Get the latest quote for a symbol directly from Alpaca API"""
    try:
        mode = 'paper' if paper else 'live'
        base_url = ALPACA_ENDPOINTS.get(mode, ALPACA_ENDPOINTS['paper'])
        endpoint = f"{base_url}/v2/stocks/{symbol}/quotes/latest"
        
        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        }
        
        response = requests.get(endpoint, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            quote = data.get('quote', {})
            
            # Midpoint price
            if quote:
                ask = float(quote.get('ap', 0))
                bid = float(quote.get('bp', 0))
                price = (ask + bid) / 2 if ask > 0 and bid > 0 else ask or bid
                
                return {
                    'symbol': symbol,
                    'price': price,
                    'bid': bid,
                    'ask': ask,
                    'timestamp': quote.get('t', 0)
                }
        
        # If we couldn't get a quote, try getting the latest trade
        endpoint = f"{base_url}/v2/stocks/{symbol}/trades/latest"
        response = requests.get(endpoint, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            trade = data.get('trade', {})
            
            if trade:
                return {
                    'symbol': symbol,
                    'price': float(trade.get('p', 0)),
                    'bid': 0,
                    'ask': 0,
                    'timestamp': trade.get('t', 0)
                }
        
        return None
    except Exception as e:
        print(f"Error fetching quote for {symbol}: {str(e)}")
        return None

def get_bars(api_key, api_secret, symbol, timeframe='1D', limit=100, paper=True, start=None, end=None):
    """
    Get price bars for a symbol directly from Alpaca API
    
    Args:
        api_key (str): Alpaca API key
        api_secret (str): Alpaca API secret
        symbol (str): Stock symbol
        timeframe (str): Bar timeframe: 1Min, 5Min, 15Min, 1H, 1D
        limit (int): Maximum number of bars to return
        paper (bool): Whether to use paper trading API
        start (str): Start date in format 'YYYY-MM-DD'
        end (str): End date in format 'YYYY-MM-DD'
    
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    try:
        mode = 'paper' if paper else 'live'
        base_url = ALPACA_ENDPOINTS.get(mode, ALPACA_ENDPOINTS['paper'])
        endpoint = f"{base_url}/v2/stocks/{symbol}/bars"
        
        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        }
        
        # Build query parameters
        params = {
            'timeframe': timeframe,
            'limit': limit,
            'adjustment': 'raw'
        }
        
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        
        response = requests.get(endpoint, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            bars = data.get('bars', [])
            
            if bars:
                # Convert to DataFrame
                df = pd.DataFrame(bars)
                
                # Rename columns to match standard OHLCV format
                df.rename(columns={
                    't': 'timestamp',
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume'
                }, inplace=True)
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Set index to timestamp
                df.set_index('timestamp', inplace=True)
                
                return df
        
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching bars for {symbol}: {str(e)}")
        return pd.DataFrame()

def get_historical_data(api_key, api_secret, symbol, timeframe='1D', start_date=None, end_date=None, period_days=100, paper=True):
    """
    Get historical price data for a symbol
    
    Args:
        api_key (str): Alpaca API key
        api_secret (str): Alpaca API secret
        symbol (str): Stock symbol
        timeframe (str): Bar timeframe: 1Min, 5Min, 15Min, 1H, 1D
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        period_days (int): Number of days of data to retrieve if start_date not provided
        paper (bool): Whether to use paper trading API
    
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    try:
        # If specific dates aren't provided, calculate based on period_days
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date and period_days:
            start_date = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d')
        
        # Get the historical data
        df = get_bars(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            timeframe=timeframe,
            limit=10000,  # Set high limit to ensure we get all data
            paper=paper,
            start=start_date,
            end=end_date
        )
        
        return df
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {str(e)}")
        return pd.DataFrame()