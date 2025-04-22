# Deploying to Streamlit Cloud

This guide provides specific instructions for deploying this trading bot to Streamlit Cloud to avoid installer errors.

## Step 1: Create a Streamlit Cloud Account
If you don't already have one, sign up at [https://streamlit.io/cloud](https://streamlit.io/cloud)

## Step 2: Prepare Your GitHub Repository
1. Create a new GitHub repository
2. Upload only the essential files to the repository:
   - The entire `streamlit_app` folder
   - `minimal_requirements.txt` (rename it to `requirements.txt` at the root level)

## Step 3: Structure Your Repository Correctly
Make sure your repository has this structure:
```
repository-root/
├── streamlit_app/
│   ├── Home.py
│   ├── pages/
│   ├── utils/
│   └── ...
└── requirements.txt  (copy from minimal_requirements.txt)
```

## Step 4: Deploy to Streamlit Cloud
1. Log in to Streamlit Cloud
2. Click "New app"
3. Connect to your GitHub repository
4. Configure the deployment:
   - Main file path: `streamlit_app/Home.py`
   - Python version: 3.9
   - Advanced settings: Add any required secrets (Alpaca API, Angel One API, etc.)

## Step 5: Troubleshooting
If you encounter installer errors:

1. Try reducing the requirements.txt file to only the absolute essentials:
```
streamlit==1.22.0
pandas==1.5.3
numpy==1.23.5
plotly==5.14.1
```

2. Check the Streamlit Cloud logs for specific error messages

3. Try deploying with just the minimal functionality first, then gradually add more features

## Step 6: Adding Secrets
For the trading bot to work properly, you'll need to add your API credentials:

1. On your app's page in Streamlit Cloud, go to "Settings" → "Secrets"
2. Add your API keys in this format:
```
ALPACA_API_KEY = "your_alpaca_key_here"
ALPACA_API_SECRET = "your_alpaca_secret_here"
ANGEL_ONE_API_KEY = "your_angel_one_key_here"
```

## Step 7: Limitations
Be aware of these limitations when deploying to Streamlit Cloud:

1. Some features may require additional setup (like TA-Lib for technical indicators)
2. Free tier has limited computational resources
3. Apps on the free tier go to sleep after inactivity

## Need More Help?
If you're still encountering issues, please refer to:

1. [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
2. [Streamlit Community Forum](https://discuss.streamlit.io/)