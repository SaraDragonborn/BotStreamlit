# Trading Bot - Python 3.12 Compatible Version

## DEPLOYMENT INSTRUCTIONS:

1. Upload ALL these files directly to the ROOT of your GitHub repository
   - Do NOT put them in a subfolder
   - All files must be at the root level

2. In Streamlit Cloud, set the main file path to: `main.py`

3. Add your API keys in Streamlit Cloud Secrets:
```
[secrets]
ALPACA_API_KEY = "your_alpaca_key"
ALPACA_API_SECRET = "your_alpaca_secret" 
ANGEL_ONE_API_KEY = "your_angel_one_key"
ANGEL_ONE_CLIENT_ID = "your_angel_one_client_id"
ANGEL_ONE_PASSWORD = "your_angel_one_password"
```

## Features

This version is compatible with Python 3.12 while maintaining most functionality:
- Portfolio watchlist
- Technical indicators
- Trading strategies
- US and Indian market trading capabilities

## Package Versions

This version uses package versions known to be compatible with Python 3.12:
- streamlit==1.27.0
- pandas==2.0.3
- numpy==1.24.3
- plotly==5.18.0
- yfinance==0.2.36
- requests==2.31.0
- scikit-learn==1.3.2
- python-dotenv==1.0.0
- matplotlib==3.7.4

Some advanced features may have reduced functionality due to compatibility constraints.
