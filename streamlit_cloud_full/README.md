# Complete Trading Bot - Streamlit Cloud Version

This is the full version of the Trading Bot Platform, adapted for compatibility with Streamlit Cloud.

## CRITICAL DEPLOYMENT STEPS:

1. Upload ALL these files directly to the ROOT of your GitHub repository
   - Do NOT put them in a subfolder
   - The requirements.txt file MUST be in the root directory
   - main.py must be in the root directory

2. In Streamlit Cloud, set the main file path to: `main.py`
   - Do NOT use "Home.py" or any other name

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

- Portfolio watchlist with up to 100 stocks
- 40+ technical indicators
- AI-powered trading strategies
- US and Indian market trading
- Comprehensive statistical data
- Day trading capabilities

## Troubleshooting

If you encounter any issues:
1. Make sure ALL files are in the ROOT directory of your repository
2. Verify that the requirements.txt file has compatible versions
3. Check that you've set the main file path to main.py
4. Ensure your API keys are added to the Streamlit Cloud Secrets
