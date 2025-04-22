# Fixing Installer Error

If you encounter this error when running the Streamlit app:
```
[12:27:28] ❗️ installer returned a non-zero exit code
[12:27:28] ❗️ Error during processing dependencies! Please fix the error and push an update, or try restarting the app.
```

## Solution 1: Install Required Dependencies

This error typically occurs when some required Python packages are missing. Try the following:

1. Make sure you have installed all required packages:
   ```bash
   pip install -r streamlit_app/requirements-minimal.txt
   ```

2. If you're on Streamlit Cloud, make sure the requirements file is properly formatted.

## Solution 2: Split Requirements Installation

Sometimes the error occurs when trying to install too many packages at once. Try installing them in smaller batches:

1. Install core dependencies first:
   ```bash
   pip install streamlit pandas numpy plotly
   ```

2. Then install additional dependencies:
   ```bash
   pip install yfinance requests
   ```

3. And finally the more specialized libraries:
   ```bash
   pip install ccxt stockstats ta-lib
   ```

## Solution 3: Use a Different Environment

If you're still facing issues:

1. Create a fresh virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies in the new environment:
   ```bash
   pip install -r streamlit_app/requirements-minimal.txt
   ```

3. Run the application:
   ```bash
   cd streamlit_app
   streamlit run Home.py
   ```

## Solution 4: Check for Conflicting Dependencies

If the error persists, it might be due to conflicting package versions:

1. Edit the requirements-minimal.txt file to use specific versions that are known to work together
2. Remove any conflicting packages or downgrade certain packages if needed

## Solution 5: Platform-Specific Solution

### For Windows:
Some packages like TA-Lib might cause issues on Windows. Try using pre-built wheels:
```bash
pip install --only-binary=:all: -r streamlit_app/requirements-minimal.txt
```

### For Linux:
On Linux systems, you might need some additional system dependencies:
```bash
sudo apt-get update
sudo apt-get install -y build-essential
```

## Solution 6: Use the Ready-to-Run Environment

For a completely hassle-free experience, consider using one of these options:

1. **Streamlit Cloud** - Deploy directly to Streamlit Cloud which has most dependencies pre-installed
2. **Google Colab** - Use Google Colab to run the app with minimal setup
3. **Docker** - Use the provided Docker setup if available

## Need Further Help?

If you continue to face issues, please open an issue with detailed error logs so we can provide specific help for your environment.