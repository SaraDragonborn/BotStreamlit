# Streamlit Cloud Deployment Guide (Updated April 2025)

This guide provides detailed instructions for deploying the full-featured AI Trading Bot to Streamlit Cloud, ensuring all functionality works correctly with Python 3.12.

## Deployment Steps

### 1. Prepare the Repository

1. Create a GitHub repository for your Streamlit application
2. Upload the entire `streamlit_v1` directory to the repository
3. Ensure the repository structure matches the following:
   ```
   streamlit_v1/
   ├── streamlit_app.py     # Main application file
   ├── requirements.txt     # Updated compatible dependencies
   ├── pages/               # Multi-page application components
   │   ├── 1_Strategies.py
   │   ├── 2_Portfolio.py
   │   ├── 3_AI_Analysis.py
   │   ├── 4_Settings.py 
   │   ├── 5_Signals_Dashboard.py
   │   └── 6_Technical_Indicators.py
   └── utils/               # Utility functions
       ├── alpaca_api.py
       ├── angel_one_api.py
       ├── credentials.py
       ├── indian_market_connector.py
       ├── indian_strategies.py
       ├── stock_info.py
       ├── stock_statistics.py
       └── strategies.py
   ```

### 2. Critical Requirements.txt Configuration

The `requirements.txt` file has been specifically configured to work with Streamlit Cloud's Python 3.12 environment. Do not modify version numbers unless absolutely necessary.

```
streamlit==1.32.0
pandas==2.0.3
numpy==1.24.3
plotly==5.18.0
yfinance==0.2.35
requests==2.31.0
python-dotenv==1.0.0
matplotlib==3.7.3
Pillow==10.1.0
scipy==1.11.4
scikit-learn==1.3.2
joblib==1.3.2
protobuf<4,>=3.20
alpaca-trade-api==3.0.2
beautifulsoup4==4.12.2
lxml==4.9.3
altair<5
toml==0.10.2
watchdog==3.0.0
jsonschema==4.20.0
pyarrow==14.0.1
```

This configuration addresses the following issues:
- Compatible numpy version with Python 3.12
- Pinned pandas version that works with the numpy version
- Specified versions for all critical dependencies
- Removed problematic packages like ccxt with Python 3.12 compatibility issues

### 3. Deploy to Streamlit Cloud

1. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click "New app"
3. Connect your GitHub repository
4. Set the main file path to `streamlit_v1/streamlit_app.py`
5. Advanced Settings:
   - Python version: 3.12.x (latest available)
   - Install dependencies from requirements.txt file
   - Set any required secrets (see section below)
6. Click "Deploy"

### 4. API Secrets Configuration

For the trading features to work properly, you'll need to configure the following secrets in Streamlit Cloud:

1. Go to your deployed app settings
2. Click on "Secrets"
3. Add the following secrets in TOML format:

```toml
[alpaca]
api_key = "YOUR_ALPACA_API_KEY"
api_secret = "YOUR_ALPACA_API_SECRET"
paper_trading = true

[angel_one]
api_key = "YOUR_ANGEL_ONE_API_KEY"
client_id = "YOUR_ANGEL_ONE_CLIENT_ID"
password = "YOUR_ANGEL_ONE_PASSWORD"
market_feed_api_key = "YOUR_ANGEL_ONE_MARKET_FEED_API_KEY"
historical_api_key = "YOUR_ANGEL_ONE_HISTORICAL_API_KEY"
```

### 5. Troubleshooting Common Issues

#### Package Compatibility Issues
If you encounter package installation errors:
1. Check Streamlit Cloud logs for specific error messages
2. Adjust package versions in requirements.txt if necessary
3. Try removing problematic packages and use alternatives when possible

#### API Connection Issues
1. Verify API credentials are correctly set in Streamlit secrets
2. Check if the API endpoints are accessible from Streamlit Cloud
3. Implement more robust error handling in the application code

#### Memory Limitations
1. Optimize data loading and processing routines
2. Implement pagination for large datasets
3. Use caching effectively with `@st.cache_data` and `@st.cache_resource`

## Maintaining Full Functionality

This deployment approach preserves all advanced features:
- Real-time market data access via Alpaca and Angel One APIs
- All technical indicators and trading strategies
- Portfolio analysis and backtesting capabilities
- Machine learning integration for strategy optimization
- Indian market support with NSE integration
- Day trading capabilities with signal processing
- Multi-currency and multi-exchange functionality

## Updates and Maintenance

When updating the application:
1. Test changes locally before pushing to GitHub
2. Monitor Streamlit Cloud logs after deployment
3. Be prepared to quickly roll back changes if issues occur
4. Consider implementing feature flags for progressive rollouts

## Security Considerations

1. Never commit API keys or secrets to your GitHub repository
2. Use Streamlit secrets management for all sensitive credentials
3. Implement proper error handling to avoid exposing sensitive information
4. Consider implementing IP restrictions if supported by your API providers

## Performance Optimization

1. Use session state efficiently to maintain state between interactions
2. Implement caching for expensive computations and API calls
3. Consider breaking complex pages into smaller components
4. Optimize data visualization for better performance

By following this guide, you'll ensure that your AI Trading Bot deploys correctly to Streamlit Cloud with all functionality intact.