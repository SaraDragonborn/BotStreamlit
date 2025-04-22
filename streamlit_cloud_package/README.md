# AI Trading Bot V1 - Streamlit Cloud Version

This is a comprehensive trading bot designed to work reliably on Streamlit Cloud with all features intact.

## Deployment Instructions

1. Upload ALL files from this folder to the ROOT of your GitHub repository
2. In Streamlit Cloud, set the main file path to: `app.py`
3. Deploy!

## Features

- Complete market dashboard with real-time data
- Strategy backtesting with multiple algorithms
- Portfolio tracking and performance analysis
- US and Indian market support through Alpaca and Angel One APIs
- API credential management
- Comprehensive settings configuration

## Adding API Keys

In Streamlit Cloud, add your API keys in the Secrets section:

```toml
[alpaca]
API_KEY = "your_alpaca_key"
API_SECRET = "your_alpaca_secret"

[angel_one]
API_KEY = "your_angel_one_key"
CLIENT_ID = "your_angel_one_client_id"
PASSWORD = "your_angel_one_password"
```

## Troubleshooting

If you encounter pip errors in Streamlit Cloud, add this to your Streamlit Cloud advanced settings:

**Command to run your app:**
```
pip install --upgrade pip && pip install -r requirements.txt && streamlit run app.py
```