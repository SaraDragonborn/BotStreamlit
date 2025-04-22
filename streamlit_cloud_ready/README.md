# Trading Bot - Streamlit Cloud Version

## Deployment Instructions

1. Create a new GitHub repository
2. Upload all these files to the ROOT of the repository
3. In Streamlit Cloud, create a new app pointing to your repository
4. Set the main file path to: `streamlit_app.py`
5. Add your API keys in Streamlit Cloud Secrets:
   ```
   [secrets]
   ALPACA_API_KEY = "your_alpaca_key"
   ALPACA_API_SECRET = "your_alpaca_secret" 
   ANGEL_ONE_API_KEY = "your_angel_one_key"
   ANGEL_ONE_CLIENT_ID = "your_angel_one_client_id"
   ANGEL_ONE_PASSWORD = "your_angel_one_password"
   ```

## Important Notes

- This version uses specific package versions compatible with Streamlit Cloud
- If you encounter any issues, try removing specific version constraints in requirements.txt
