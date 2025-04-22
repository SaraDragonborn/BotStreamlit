# Complete Trading Bot Platform for Streamlit Cloud

This is the full version of the Trading Bot Platform configured for Streamlit Cloud deployment.

## Features

- US and Indian stock market trading
- Portfolio watchlist with up to 100 stocks
- 40+ technical indicators
- AI-powered trading strategies
- Day trading capabilities
- Comprehensive statistical data
- Multi-exchange support

## Deployment Instructions

1. Upload this entire folder to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect to your GitHub repository
4. Set the main file path to `Home.py`
5. Add your API keys in the Secrets section

## Required API Keys

Add these to your Streamlit Cloud Secrets:

```
[secrets]
ALPACA_API_KEY = "your_alpaca_key_here" 
ALPACA_API_SECRET = "your_alpaca_secret_here"
ANGEL_ONE_API_KEY = "your_angel_one_key_here"
ANGEL_ONE_CLIENT_ID = "your_angel_one_client_id_here"
ANGEL_ONE_PASSWORD = "your_angel_one_password_here"
```

## Troubleshooting

If you encounter any installation issues, please check the compatibility of package versions in requirements.txt. Streamlit Cloud may require specific version combinations.
