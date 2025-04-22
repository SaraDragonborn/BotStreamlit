# Trading Bot Platform - IMPORTANT INSTRUCTIONS

## CRITICAL DEPLOYMENT STEPS:

1. Upload ALL these files to the ROOT directory of your GitHub repository.
   DO NOT put them in a subfolder!

2. In Streamlit Cloud, set the main file path to simply: `main.py`

3. The `requirements.txt` file MUST be in the ROOT directory of your repository.

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

If you continue to experience installation issues, try:
1. Make sure ALL files are in the ROOT directory
2. Verify that requirements.txt is directly visible at the repository root
3. Start with a minimal set of requirements and gradually add more
