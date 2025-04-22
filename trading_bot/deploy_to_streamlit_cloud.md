# Deploying to Streamlit Cloud

This guide provides step-by-step instructions for deploying your AI Trading Bot to Streamlit Cloud for easy access anywhere.

## Prerequisites

1. A [GitHub](https://github.com) account
2. A [Streamlit Cloud](https://streamlit.io/cloud) account (free tier available)

## Preparation Steps

### 1. Create a GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Clone the repository to your local machine or use GitHub's "Upload Files" feature
3. Upload all files from the `streamlit_app` directory to your repository
   - Make sure to include the `requirements.txt` file
   - Include all Python files, utility modules, and data directories

### 2. Prepare for Deployment

1. Make sure your GitHub repository contains:
   - `Home.py` (main entry point)
   - `pages/` directory with all page files
   - `utils/` directory with utility functions
   - `requirements.txt` file

2. Verify that your `requirements.txt` includes all necessary dependencies:
   ```
   streamlit==1.44.0
   pandas==2.1.4
   numpy==1.26.3
   plotly==6.0.0
   requests==2.31.0
   python-dotenv==1.0.0
   yfinance==0.2.36
   matplotlib==3.8.2
   ccxt==4.1.22
   nltk==3.8.1
   scikit-learn==1.3.2
   torch==2.1.2
   stockstats==0.5.4
   alpaca-trade-api==3.0.1
   python-dateutil==2.8.2
   ```

## Deployment Steps

### 1. Connect to Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in
2. Click "New App" to create a new application
3. Connect to your GitHub repository:
   - Select the repository from the dropdown
   - Select the main branch (usually `main` or `master`)
   - Set "Main file path" to `Home.py`

### 2. Configure Environment Secrets

1. In the Streamlit deployment settings, go to "Secrets"
2. Add your API keys:
   ```
   ALPACA_API_KEY = "your_alpaca_api_key"
   ALPACA_API_SECRET = "your_alpaca_api_secret"
   ```
3. Add any other required API keys or secrets

### 3. Advanced Settings (Optional)

1. Set Python version (if needed) - click "Advanced settings" 
2. Configure memory requirements if your app needs more resources
3. Set up custom subdomains if you have a Streamlit Teams account

### 4. Deploy

1. Click "Deploy" to start the deployment process
2. Wait for the build and deployment process to complete
3. Your app will be available at `https://yourappname.streamlit.app`

## Updating Your Deployment

When you make changes to your code:

1. Push the changes to your GitHub repository
2. Streamlit Cloud will automatically detect the changes and redeploy your app

## Troubleshooting

1. **App crashes on startup**: Check the logs in Streamlit Cloud dashboard for errors
2. **Missing dependencies**: Make sure all required packages are in `requirements.txt`
3. **API connection issues**: Verify your API keys in the Secrets management section
4. **Memory errors**: Consider upgrading your Streamlit plan or optimizing your code

## Best Practices

1. **Security**: Never commit API keys to your repository
2. **Performance**: Optimize data loading and caching for better performance
3. **User Experience**: Use st.cache and other performance techniques
4. **Monitoring**: Regularly check your app's logs and performance

## Additional Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)
- [GitHub Actions for CI/CD](https://docs.github.com/en/actions)