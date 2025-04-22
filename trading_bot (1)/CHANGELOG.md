# AI Trading Bot Changelog

## Version 1.1.0 (2025-04-22)

### Major API Connection Updates
- **Direct API Connection**: Completely redesigned API connectivity to use direct connections to Alpaca API
- **Fixed Authentication Issues**: Properly passing API keys in request headers
- **Improved Error Handling**: Better error messages for connection failures

### New Features
- **Enhanced Order Management**: Added direct order placement and management functionality
- **Trading Mode Selection**: Support for both paper and live trading modes with proper API endpoint selection
- **Streamlined UI**: Improved interface design with better feedback on API connection status

### Bug Fixes
- Fixed issue where API connection test failed after saving credentials
- Fixed positions not showing correctly in Portfolio page
- Fixed error when placing orders in non-US market hours

## Version 1.0.0 (2025-03-15)

### Initial Release
- Streamlit-based UI for AI trading
- Alpaca API integration 
- Paper trading support
- Strategy creation and backtesting
- Portfolio monitoring and visualization
- AI analysis with FinGPT, DeepTradeBot, and FinRL
- News sentiment analysis
- Trading signals generation