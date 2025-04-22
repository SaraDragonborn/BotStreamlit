"""
Utilities for fetching stock statistics and analytics
"""
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# Get basic stock statistics for a symbol
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
    try:
        # Use US or Indian market data sources based on exchange
        if exchange.upper() in ["NSE", "BSE"]:
            # For Indian stocks, append "-EQ" if not already present for NSE stocks
            if exchange.upper() == "NSE" and not symbol.endswith("-EQ") and not symbol.startswith("NIFTY") and not symbol.startswith("BANK"):
                symbol = f"{symbol}-EQ"
            return get_indian_stock_statistics(symbol, exchange)
        else:
            # Use yfinance for US stocks
            return get_us_stock_statistics(symbol)
    except Exception as e:
        st.error(f"Error fetching statistics for {symbol}: {str(e)}")
        return None

def get_us_stock_statistics(symbol):
    """Get statistics for US stocks using yfinance"""
    try:
        # Get stock info from yfinance
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Get historical data for calculations
        hist = stock.history(period="1y")
        
        if hist.empty:
            return None
        
        # Calculate volatility and other metrics
        returns = hist['Close'].pct_change().dropna()
        
        # Calculate statistics
        stats = {
            'symbol': symbol,
            'name': info.get('shortName', 'Unknown'),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'exchange': info.get('exchange', 'Unknown'),
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
            'day_change': info.get('regularMarketChangePercent', 0),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'eps': info.get('trailingEps', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'volume': info.get('volume', 0),
            'avg_volume': info.get('averageVolume', 0),
            'price_to_book': info.get('priceToBook', 0),
            'beta': info.get('beta', 0),
            '52w_high': info.get('fiftyTwoWeekHigh', 0),
            '52w_low': info.get('fiftyTwoWeekLow', 0),
            'analyst_rating': info.get('recommendationKey', 'N/A'),
            'target_price': info.get('targetMeanPrice', 0)
        }
        
        # Calculate additional metrics
        if not hist.empty and len(hist) > 20:
            # Volatility (annualized standard deviation of returns)
            stats['volatility'] = returns.std() * np.sqrt(252) * 100
            
            # RSI (Relative Strength Index)
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            stats['rsi'] = 100 - (100 / (1 + rs.iloc[-1])) if loss.iloc[-1] != 0 else 50
            
            # 50-day and 200-day Moving Averages
            stats['ma_50'] = hist['Close'].rolling(window=50).mean().iloc[-1]
            stats['ma_200'] = hist['Close'].rolling(window=200).mean().iloc[-1]
            
            # Calculate z-scores for current price relative to moving averages
            stats['ma_50_z'] = (stats['current_price'] - stats['ma_50']) / (hist['Close'].rolling(window=50).std().iloc[-1])
            stats['ma_200_z'] = (stats['current_price'] - stats['ma_200']) / (hist['Close'].rolling(window=200).std().iloc[-1])
            
            # 52-week percentile (where current price is relative to 52-week range)
            range_52w = stats['52w_high'] - stats['52w_low']
            if range_52w > 0:
                stats['52w_percentile'] = (stats['current_price'] - stats['52w_low']) / range_52w * 100
            else:
                stats['52w_percentile'] = 50
            
            # 20-day ATR (Average True Range)
            high_low = hist['High'] - hist['Low']
            high_close = abs(hist['High'] - hist['Close'].shift())
            low_close = abs(hist['Low'] - hist['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            stats['atr_20'] = tr.rolling(window=20).mean().iloc[-1]
        
        return stats
    except Exception as e:
        st.error(f"Error fetching US stock statistics for {symbol}: {str(e)}")
        return None

def get_indian_stock_statistics(symbol, exchange="NSE"):
    """Get statistics for Indian stocks"""
    try:
        # For Indian stocks, we need to use appropriate data sources
        # This is a placeholder - you would integrate with Angel One API here
        # For now, we'll generate reasonable values
        
        # Get base data from yfinance using the correct Indian exchange symbol format
        if exchange.upper() == "NSE":
            yf_symbol = f"{symbol.replace('-EQ', '')}.NS"
        else:  # BSE
            yf_symbol = f"{symbol.replace('-EQ', '')}.BO"
        
        try:
            stock = yf.Ticker(yf_symbol)
            info = stock.info
            hist = stock.history(period="1y")
        except:
            # Fallback for indices
            if "NIFTY" in symbol or "BANK" in symbol:
                yf_symbol = "^NSEI" if "NIFTY" in symbol else "^BSESN"
                stock = yf.Ticker(yf_symbol)
                info = stock.info
                hist = stock.history(period="1y")
            else:
                # For stocks that can't be found, generate reasonable values
                # This would be replaced with actual API calls in production
                return get_placeholder_indian_statistics(symbol, exchange)
        
        # Extract or calculate statistics
        if hist.empty:
            return get_placeholder_indian_statistics(symbol, exchange)
        
        # Calculate returns for additional metrics
        returns = hist['Close'].pct_change().dropna()
        
        stats = {
            'symbol': symbol,
            'name': info.get('shortName', symbol),
            'sector': info.get('sector', 'Indian Equity'),
            'industry': info.get('industry', 'N/A'),
            'exchange': exchange,
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', hist['Close'].iloc[-1])),
            'day_change': info.get('regularMarketChangePercent', 0),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'eps': info.get('trailingEps', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'volume': info.get('volume', 0),
            'avg_volume': info.get('averageVolume', 0),
            'price_to_book': info.get('priceToBook', 0),
            'beta': info.get('beta', 0),
            '52w_high': info.get('fiftyTwoWeekHigh', hist['High'].max()),
            '52w_low': info.get('fiftyTwoWeekLow', hist['Low'].min()),
            'analyst_rating': info.get('recommendationKey', 'N/A'),
            'target_price': info.get('targetMeanPrice', 0)
        }
        
        # Calculate additional metrics
        if not hist.empty and len(hist) > 20:
            # Volatility (annualized standard deviation of returns)
            stats['volatility'] = returns.std() * np.sqrt(252) * 100
            
            # RSI (Relative Strength Index)
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            stats['rsi'] = 100 - (100 / (1 + rs.iloc[-1])) if loss.iloc[-1] != 0 else 50
            
            # 50-day and 200-day Moving Averages
            stats['ma_50'] = hist['Close'].rolling(window=50).mean().iloc[-1]
            stats['ma_200'] = hist['Close'].rolling(window=200).mean().iloc[-1]
            
            # Calculate z-scores for current price relative to moving averages
            stats['ma_50_z'] = (stats['current_price'] - stats['ma_50']) / (hist['Close'].rolling(window=50).std().iloc[-1])
            stats['ma_200_z'] = (stats['current_price'] - stats['ma_200']) / (hist['Close'].rolling(window=200).std().iloc[-1])
            
            # 52-week percentile
            range_52w = stats['52w_high'] - stats['52w_low']
            if range_52w > 0:
                stats['52w_percentile'] = (stats['current_price'] - stats['52w_low']) / range_52w * 100
            else:
                stats['52w_percentile'] = 50
            
            # 20-day ATR (Average True Range)
            high_low = hist['High'] - hist['Low']
            high_close = abs(hist['High'] - hist['Close'].shift())
            low_close = abs(hist['Low'] - hist['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            stats['atr_20'] = tr.rolling(window=20).mean().iloc[-1]
        
        return stats
    except Exception as e:
        st.error(f"Error fetching Indian stock statistics for {symbol}: {str(e)}")
        return get_placeholder_indian_statistics(symbol, exchange)

def get_placeholder_indian_statistics(symbol, exchange):
    """Generate placeholder statistics for Indian stocks when data isn't available"""
    # This is only used when actual data can't be retrieved
    # In a real implementation, this would be replaced with actual Angel One API calls
    return {
        'symbol': symbol,
        'name': symbol.replace('-EQ', ''),
        'sector': 'Indian Equity',
        'industry': 'N/A',
        'exchange': exchange,
        'current_price': 1000.0,  # Placeholder price
        'day_change': 0.0,
        'market_cap': 1000000000,
        'pe_ratio': 15.0,
        'eps': 65.0,
        'dividend_yield': 2.0,
        'volume': 500000,
        'avg_volume': 750000,
        'price_to_book': 2.5,
        'beta': 1.0,
        '52w_high': 1200.0,
        '52w_low': 800.0,
        'analyst_rating': 'N/A',
        'target_price': 1100.0,
        'volatility': 25.0,
        'rsi': 50.0,
        'ma_50': 980.0,
        'ma_200': 950.0,
        'ma_50_z': 0.2,
        'ma_200_z': 0.5,
        '52w_percentile': 50.0,
        'atr_20': 20.0
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
    results = []
    
    # Convert exchange to list if it's a single string
    exchanges = [exchange] * len(symbols) if isinstance(exchange, str) else exchange
    
    # Create list of symbol-exchange pairs
    tasks = list(zip(symbols, exchanges))
    
    # Process in batches to avoid rate limiting
    batch_size = 10
    all_stats = []
    
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        
        with ThreadPoolExecutor(max_workers=min(batch_size, len(batch))) as executor:
            future_to_symbol = {executor.submit(get_stock_statistics, symbol, exch): symbol for symbol, exch in batch}
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    stats = future.result()
                    if stats:
                        all_stats.append(stats)
                except Exception as e:
                    st.error(f"Error processing {symbol}: {str(e)}")
        
        # Add a small delay between batches to avoid rate limiting
        if i + batch_size < len(tasks):
            time.sleep(1)
    
    return all_stats

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
    
    df = pd.DataFrame(stats_list)
    
    # Reorder columns for better presentation
    preferred_order = [
        'symbol', 'name', 'exchange', 'current_price', 'day_change',
        'rsi', 'volatility', '52w_percentile', 'ma_50_z', 'ma_200_z',
        'market_cap', 'pe_ratio', 'eps', 'dividend_yield', 
        'volume', 'avg_volume', 'price_to_book', 'beta',
        '52w_high', '52w_low', 'analyst_rating', 'target_price',
        'sector', 'industry'
    ]
    
    # Add any remaining columns not in preferred_order
    remaining_cols = [col for col in df.columns if col not in preferred_order]
    column_order = preferred_order + remaining_cols
    
    # Only include columns that exist in the DataFrame
    final_columns = [col for col in column_order if col in df.columns]
    
    return df[final_columns]

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
    
    # Make a copy to avoid modifying the original
    formatted_df = df.copy()
    
    # Format numerical columns
    if 'current_price' in formatted_df.columns:
        formatted_df['current_price'] = formatted_df['current_price'].apply(lambda x: f"${x:.2f}")
    
    if 'day_change' in formatted_df.columns:
        formatted_df['day_change'] = formatted_df['day_change'].apply(lambda x: f"{x:+.2f}%")
    
    if 'market_cap' in formatted_df.columns:
        formatted_df['market_cap'] = formatted_df['market_cap'].apply(lambda x: f"${x/1e9:.2f}B" if x >= 1e9 else f"${x/1e6:.2f}M")
    
    if 'pe_ratio' in formatted_df.columns:
        formatted_df['pe_ratio'] = formatted_df['pe_ratio'].apply(lambda x: f"{x:.2f}")
    
    if 'dividend_yield' in formatted_df.columns:
        formatted_df['dividend_yield'] = formatted_df['dividend_yield'].apply(lambda x: f"{x:.2f}%")
    
    if 'volume' in formatted_df.columns:
        formatted_df['volume'] = formatted_df['volume'].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.2f}K")
    
    if 'avg_volume' in formatted_df.columns:
        formatted_df['avg_volume'] = formatted_df['avg_volume'].apply(lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.2f}K")
    
    if 'beta' in formatted_df.columns:
        formatted_df['beta'] = formatted_df['beta'].apply(lambda x: f"{x:.2f}")
    
    if '52w_high' in formatted_df.columns:
        formatted_df['52w_high'] = formatted_df['52w_high'].apply(lambda x: f"${x:.2f}")
    
    if '52w_low' in formatted_df.columns:
        formatted_df['52w_low'] = formatted_df['52w_low'].apply(lambda x: f"${x:.2f}")
    
    if 'target_price' in formatted_df.columns:
        formatted_df['target_price'] = formatted_df['target_price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
    
    if 'volatility' in formatted_df.columns:
        formatted_df['volatility'] = formatted_df['volatility'].apply(lambda x: f"{x:.2f}%")
    
    if 'rsi' in formatted_df.columns:
        formatted_df['rsi'] = formatted_df['rsi'].apply(lambda x: f"{x:.2f}")
    
    # Rename columns for better display
    column_renames = {
        'symbol': 'Symbol',
        'name': 'Name',
        'exchange': 'Exchange',
        'current_price': 'Price',
        'day_change': 'Day Chg',
        'market_cap': 'Mkt Cap',
        'pe_ratio': 'P/E',
        'eps': 'EPS',
        'dividend_yield': 'Div Yield',
        'volume': 'Volume',
        'avg_volume': 'Avg Vol',
        'price_to_book': 'P/B',
        'beta': 'Beta',
        '52w_high': '52W High',
        '52w_low': '52W Low',
        'analyst_rating': 'Rating',
        'target_price': 'Target',
        'sector': 'Sector',
        'industry': 'Industry',
        'volatility': 'Volatility',
        'rsi': 'RSI',
        'ma_50': '50-day MA',
        'ma_200': '200-day MA',
        'ma_50_z': '50d Z-Score',
        'ma_200_z': '200d Z-Score',
        '52w_percentile': '52W %ile',
        'atr_20': '20d ATR'
    }
    
    # Only rename columns that exist in the DataFrame
    rename_dict = {k: v for k, v in column_renames.items() if k in formatted_df.columns}
    return formatted_df.rename(columns=rename_dict)