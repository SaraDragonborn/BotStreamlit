# Deploying AI Trading Bot to Streamlit Cloud

This guide provides detailed instructions for deploying your AI Trading Bot to Streamlit Cloud when there are issues with the Streamlit secrets management.

## Method 1: Using Environment Variables

Streamlit Cloud allows you to set environment variables through the web interface:

1. Deploy your app to Streamlit Cloud
2. Go to your app settings
3. Find the "Secrets" section
4. Add your Alpaca API keys as environment variables:
   ```
   ALPACA_API_KEY=your_api_key_here
   ALPACA_API_SECRET=your_api_secret_here
   ```

## Method 2: Direct Input in the App

If Method 1 doesn't work, the app is already designed to allow direct input of API keys:

1. Deploy your app to Streamlit Cloud
2. When the app loads, enter your Alpaca API keys in the sidebar
3. Click "Save Credentials" to store them in session state
4. Click "Test Connection" to verify they work

## Troubleshooting Deployment Issues

If you encounter installation errors when deploying to Streamlit Cloud:

1. Check the deployment logs for specific error messages

2. Try using the minimal requirements:
   - Rename `requirements-minimal.txt` to `requirements.txt`
   - Deploy again

3. Use packages.txt for system dependencies:
   - Make sure `packages.txt` is at the root of your repository
   - It should include necessary system dependencies

4. Set Python version:
   - In the Streamlit Cloud app settings, try setting Python to version 3.9 or 3.10

5. Remove problematic dependencies:
   - If a specific package is causing issues, you can comment it out in requirements.txt
   - Some AI features may not work but core trading will still function

For more detailed troubleshooting information, see `streamlit_cloud_fixes.md`.