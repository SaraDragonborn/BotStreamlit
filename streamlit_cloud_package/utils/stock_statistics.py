"""
Utilities for fetching stock statistics and analytics
"""
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import requests
import json
from datetime import datetime, timedelta
import time

def get_stock_statistics(symbol, exchange="US"):
    """
    Get comprehensive statistics for a stock
    
    Parameters:
    -----------
    symbol : str
        Stock symbol
    exchange : str
        Exchange code (default "US")
        
    Returns:
    --------
    dict or None
        Dictionary with stock statistics or None if not found
    """
    if exchange == "US":
        return get_us_stock_statistics(symbol)
    elif exchange in ["NSE", "BSE"]:
        return get_indian_stock_statistics(symbol, exchange)
    else:
        return None

def get_us_stock_statistics(symbol):
    """Get statistics for US stocks using yfinance"""
    try:
        # Get ticker info
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get historical data for calculations
        hist = ticker.history(period="1y")
        
        if hist.empty or not info:
            return None
        
        # Calculate statistics
        current_price = hist['Close'].iloc[-1] if not hist.empty else None
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else None
        change = ((current_price - prev_close) / prev_close * 100) if prev_close and current_price else None
        
        # 52-week high/low
        high_52week = hist['High'].max() if not hist.empty else None
        low_52week = hist['Low'].min() if not hist.empty else None
        
        # Calculate volatility (annualized standard deviation)
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100 if not returns.empty else None
        
        # Calculate average volume
        avg_volume = hist['Volume'].mean() if not hist.empty else None
        
        # Beta calculation (if available)
        beta = info.get('beta', None)
        
        # P/E ratio
        pe_ratio = info.get('trailingPE', None)
        
        # Market cap
        market_cap = info.get('marketCap', None)
        
        # Create statistics dictionary
        stats = {
            'symbol': symbol,
            'name': info.get('shortName', symbol),
            'exchange': info.get('exchange', 'US'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'current_price': current_price,
            'change_percent': change,
            'day_high': hist['High'].iloc[-1] if not hist.empty else None,
            'day_low': hist['Low'].iloc[-1] if not hist.empty else None,
            'open': hist['Open'].iloc[-1] if not hist.empty else None,
            'prev_close': prev_close,
            'volume': hist['Volume'].iloc[-1] if not hist.empty else None,
            'avg_volume': avg_volume,
            'market_cap': market_cap,
            'pe_ratio': pe_ratio,
            'eps': info.get('trailingEPS', None),
            'dividend_yield': info.get('dividendYield', None) * 100 if info.get('dividendYield') else None,
            'beta': beta,
            'high_52week': high_52week,
            'low_52week': low_52week,
            'volatility': volatility,
        }
        
        return stats
    except Exception as e:
        print(f"Error fetching statistics for {symbol}: {e}")
        return None

def get_indian_stock_statistics(symbol, exchange="NSE"):
    """Get statistics for Indian stocks"""
    try:
        # For Indian stocks, we need to format the ticker symbol correctly
        if exchange == "NSE":
            ticker_symbol = f"{symbol}.NS"
        else:  # BSE
            ticker_symbol = f"{symbol}.BO"
        
        # Get ticker info
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Get historical data for calculations
        hist = ticker.history(period="1y")
        
        if hist.empty or not info:
            # Fallback to placeholder data if real data isn't available
            return get_placeholder_indian_statistics(symbol, exchange)
        
        # Calculate statistics (same as US stocks)
        current_price = hist['Close'].iloc[-1] if not hist.empty else None
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else None
        change = ((current_price - prev_close) / prev_close * 100) if prev_close and current_price else None
        
        # 52-week high/low
        high_52week = hist['High'].max() if not hist.empty else None
        low_52week = hist['Low'].min() if not hist.empty else None
        
        # Calculate volatility (annualized standard deviation)
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100 if not returns.empty else None
        
        # Calculate average volume
        avg_volume = hist['Volume'].mean() if not hist.empty else None
        
        # Beta calculation (if available)
        beta = info.get('beta', None)
        
        # Create statistics dictionary
        stats = {
            'symbol': symbol,
            'name': info.get('shortName', symbol),
            'exchange': exchange,
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'current_price': current_price,
            'change_percent': change,
            'day_high': hist['High'].iloc[-1] if not hist.empty else None,
            'day_low': hist['Low'].iloc[-1] if not hist.empty else None,
            'open': hist['Open'].iloc[-1] if not hist.empty else None,
            'prev_close': prev_close,
            'volume': hist['Volume'].iloc[-1] if not hist.empty else None,
            'avg_volume': avg_volume,
            'market_cap': info.get('marketCap', None),
            'pe_ratio': info.get('trailingPE', None),
            'eps': info.get('trailingEPS', None),
            'dividend_yield': info.get('dividendYield', None) * 100 if info.get('dividendYield') else None,
            'beta': beta,
            'high_52week': high_52week,
            'low_52week': low_52week,
            'volatility': volatility,
        }
        
        return stats
    except Exception as e:
        print(f"Error fetching statistics for {symbol}: {e}")
        return get_placeholder_indian_statistics(symbol, exchange)

def get_placeholder_indian_statistics(symbol, exchange):
    """Generate placeholder statistics for Indian stocks when data isn't available"""
    # This should be called only if API data retrieval fails
    # In a real implementation, you would use more accurate data sources
    # In this case, we'll return a structured dictionary with None values
    return {
        'symbol': symbol,
        'name': f"{symbol}",
        'exchange': exchange,
        'sector': 'N/A',
        'industry': 'N/A',
        'current_price': None,
        'change_percent': None,
        'day_high': None,
        'day_low': None,
        'open': None,
        'prev_close': None,
        'volume': None,
        'avg_volume': None,
        'market_cap': None,
        'pe_ratio': None,
        'eps': None,
        'dividend_yield': None,
        'beta': None,
        'high_52week': None,
        'low_52week': None,
        'volatility': None,
        'error': 'Data temporarily unavailable. Please check your API connection.'
    }

def batch_get_statistics(symbols, exchange="US"):
    """
    Get statistics for multiple symbols in parallel
    
    Parameters:
    -----------
    symbols : list
        List of stock symbols
    exchange : str or list
        Exchange code (default "US") or list of exchanges matching symbols
        
    Returns:
    --------
    list
        List of statistics dictionaries for each symbol
    """
    # Process exchanges parameter
    if isinstance(exchange, str):
        exchanges = [exchange] * len(symbols)
    else:
        exchanges = exchange
        
    # Use ThreadPoolExecutor to parallelize API calls
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks
        futures = [executor.submit(get_stock_statistics, symbol, exc) 
                  for symbol, exc in zip(symbols, exchanges)]
        
        # Collect results
        results = []
        for future in futures:
            result = future.result()
            if result is not None:
                results.append(result)
        
    return results

def create_statistics_df(stats_list):
    """
    Convert a list of statistics dictionaries to a DataFrame
    
    Parameters:
    -----------
    stats_list : list
        List of statistics dictionaries
        
    Returns:
    --------
    DataFrame
        DataFrame with stock statistics
    """
    if not stats_list:
        return pd.DataFrame()
    
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(stats_list)
    
    # Ensure consistent types
    numeric_cols = ['current_price', 'change_percent', 'day_high', 'day_low', 
                    'open', 'prev_close', 'volume', 'avg_volume', 'market_cap',
                    'pe_ratio', 'eps', 'dividend_yield', 'beta', 
                    'high_52week', 'low_52week', 'volatility']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def format_statistics_df(df):
    """
    Format a statistics DataFrame for display
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with stock statistics
        
    Returns:
    --------
    DataFrame
        Formatted DataFrame
    """
    if df.empty:
        return df
    
    # Create a copy of the DataFrame
    formatted_df = df.copy()
    
    # Format numeric columns
    if 'current_price' in formatted_df.columns:
        formatted_df['current_price'] = formatted_df['current_price'].apply(
            lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
    
    if 'change_percent' in formatted_df.columns:
        formatted_df['change_percent'] = formatted_df['change_percent'].apply(
            lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
    
    if 'day_high' in formatted_df.columns:
        formatted_df['day_high'] = formatted_df['day_high'].apply(
            lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
    
    if 'day_low' in formatted_df.columns:
        formatted_df['day_low'] = formatted_df['day_low'].apply(
            lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
    
    if 'volume' in formatted_df.columns:
        formatted_df['volume'] = formatted_df['volume'].apply(
            lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
    
    if 'market_cap' in formatted_df.columns:
        formatted_df['market_cap'] = formatted_df['market_cap'].apply(
            lambda x: f"${x/1e9:.2f}B" if pd.notnull(x) and x > 1e9 else
                     (f"${x/1e6:.2f}M" if pd.notnull(x) else "N/A"))
    
    if 'pe_ratio' in formatted_df.columns:
        formatted_df['pe_ratio'] = formatted_df['pe_ratio'].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    
    if 'dividend_yield' in formatted_df.columns:
        formatted_df['dividend_yield'] = formatted_df['dividend_yield'].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    
    if 'volatility' in formatted_df.columns:
        formatted_df['volatility'] = formatted_df['volatility'].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    
    return formatted_df