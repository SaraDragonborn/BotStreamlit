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

def get_market_data():
    """Get market data for major indices via a free API"""
    try:
        # Sample data structure for major indices
        market_indices = {
            'S&P 500': {'price': 4700.25, 'change_percent': 0.75},
            'Dow Jones': {'price': 37100.50, 'change_percent': 0.45},
            'Nasdaq': {'price': 16550.80, 'change_percent': 1.25},
            'Russell 2000': {'price': 2080.20, 'change_percent': -0.32}
        }
        
        # In a real implementation, we would fetch actual data from a financial API
        # For demo purposes, we're returning this static data
        return market_indices
    except Exception as e:
        print(f"Error fetching market data: {str(e)}")
        return {}