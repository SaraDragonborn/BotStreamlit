# Portfolio Watchlist Feature

## Overview

The Portfolio page now includes a powerful Watchlist feature that allows users to track up to 100 stocks with detailed statistics. The watchlist displays comprehensive data for each stock, prominently showing current price, day high/low, and volume information to help with decision making.

## Key Features

1. **Track Up to 100 Stocks**: Monitor a large collection of stocks across different exchanges
2. **Multi-Exchange Support**: Add symbols from US markets, NSE (India), and BSE (India)
3. **Comprehensive Statistics**: View important metrics like:
   - Current price
   - Day high and low values
   - Volume information
   - Technical indicators (RSI, Moving Averages)
   - Volatility metrics
   - 52-week range information

4. **Bulk Symbol Addition**: Add multiple symbols at once for efficient watchlist creation
5. **Filtering & Sorting**: Filter by exchange and sort by various metrics
6. **Stock Comparison**: Compare multiple stocks side-by-side using custom metrics
7. **Quick Actions**: Trade or analyze directly from the watchlist

## How to Use

1. **Add Symbols**:
   - Enter a symbol in the "Add Symbol to Watchlist" field
   - Select the appropriate exchange (US, NSE, BSE)
   - Click "Add Symbol"

2. **Bulk Add**:
   - Click "Bulk Add Symbols" expander
   - Enter multiple symbols (one per line)
   - Select the exchange
   - Click "Add All Symbols"

3. **View and Filter**:
   - Use the "Filter by Exchange" option to focus on specific exchanges
   - Sort by different metrics using "Sort By" dropdown
   - Choose ascending or descending order

4. **Work with Stocks**:
   - Select a symbol to see detailed information
   - Use Quick Actions (Trade, Analyze, Remove) for selected stocks
   - View technical indicators summary for the selected stock

5. **Compare Stocks**:
   - Use the "Compare Stocks" expander
   - Select multiple stocks to compare
   - Choose a comparison metric
   - View the side-by-side comparison chart and table

## Implementation Details

The watchlist feature leverages:
- Real-time market data via APIs (US markets and Indian markets)
- Technical analysis calculations for indicators
- Session state for persistent watchlist between page refreshes
- Caching for efficient data retrieval
- Interactive charts and tables for data visualization

## File Changes

1. `streamlit_app/pages/2_Portfolio.py`: Updated with new Watchlist tab
2. `streamlit_app/utils/stock_statistics.py`: New module for fetching and calculating stock statistics

## Integration with Indian Market

The watchlist fully supports Indian market symbols (NSE/BSE) alongside US market symbols, allowing for a comprehensive view of a global portfolio. Users can add symbols like:

- US Market: AAPL, MSFT, GOOGL
- NSE: RELIANCE-EQ, TCS-EQ, HDFCBANK-EQ
- Indices: NIFTY 50, BANKNIFTY