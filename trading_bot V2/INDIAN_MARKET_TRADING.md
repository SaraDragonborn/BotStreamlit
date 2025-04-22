# Indian Market Trading with Angel One API

## Overview

This feature enables trading on the Indian stock market (NSE/BSE) using the Angel One API integration. The functionality allows for:

- Trading Indian stocks (RELIANCE-EQ, TCS-EQ, etc.)
- Trading on the Nifty and BankNifty indices
- Using India-specific strategies optimized for the NSE/BSE markets
- Getting real-time and historical data from Indian exchanges

## Setup and Configuration

### Prerequisites

To use the Indian market trading functionality, you need to have an Angel One trading account and API access with the following credentials:

1. **Angel One Client ID**: Your Angel One client/user ID
2. **Angel One Password**: Your Angel One account password
3. **Angel One API Key**: For trading operations
4. **Angel One Historical API Key**: For historical data access
5. **Angel One Market Feed API Key**: For real-time market data feeds

### Configuration

1. Navigate to the **Settings** page of the application
2. Go to the **API Connections** tab
3. Scroll to the **Angel One API Settings** section
4. Enter your credentials
5. Click "Test Angel One Connection" to verify your setup
6. Click "Save Angel One Settings" to store your credentials

## Available Features

### Indian Market Symbols

The following Indian market symbols are available by default:

- **Indices**: NIFTY 50, BANKNIFTY
- **Stocks**: RELIANCE-EQ, TCS-EQ, HDFCBANK-EQ, INFY-EQ, ICICIBANK-EQ, KOTAKBANK-EQ, HINDUNILVR-EQ, SBIN-EQ, BAJFINANCE-EQ, BHARTIARTL-EQ, ITC-EQ, ASIANPAINT-EQ, MARUTI-EQ, TITAN-EQ

You can also search for additional symbols in the Strategy Builder.

### Indian Market Strategies

The following Indian market specific strategies are available:

1. **Nifty/Bank Nifty Momentum**: Uses EMA crossovers and RSI for signal generation optimized for Indian index movements.

2. **Option Writing Strategy**: Specifically designed for NSE F&O, used for weekly or monthly expiry options strategies with IV considerations.

3. **VWAP NSE Intraday**: VWAP-based intraday strategy specifically for NSE stocks using Volume Weighted Average Price with deviation bands.

4. **Nifty Gap Trading**: Gap trading strategy for Nifty, specifically designed for Indian market opening gaps.

### Running Backtests

To run a backtest on an Indian market symbol:

1. Go to the **Strategies** page
2. Select or create a strategy with an Indian market type
3. Select an Indian market symbol
4. Go to the **Backtesting** tab
5. Configure your backtest parameters
6. Click "Run Backtest"

## Testing the Integration

A test script is provided to verify the functionality of the Indian market integration:

```bash
./run_indian_market_test.sh
```

This script tests:
- Connection to Angel One API
- Retrieval of Indian market symbols
- Fetching historical data
- Running backtests with Indian market strategies

## Limitations

- **Trading Hours**: NSE/BSE markets operate from 9:15 AM to 3:30 PM IST, Monday-Friday
- **Data Availability**: Historical data availability may vary depending on your Angel One subscription
- **Holiday Calendar**: Indian markets have different holidays which are accounted for in the trading calendar

## Troubleshooting

If you encounter issues with the Indian market functionality:

1. **Authentication Errors**: Verify your Angel One credentials in the Settings page
2. **No Data Available**: Check if the symbol exists and trading is active for that stock/index
3. **Rate Limiting**: Angel One API has rate limits - avoid excessive API calls
4. **Connection Issues**: Check your internet connection and firewall settings

For further assistance, please contact Angel One support for API-specific issues or check the application logs for detailed error messages.