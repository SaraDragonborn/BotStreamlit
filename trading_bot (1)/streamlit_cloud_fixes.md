# Fixing Deployment Issues on Streamlit Cloud

If you encounter the error `installer returned a non-zero exit code` when deploying to Streamlit Cloud, here are some solutions:

## Solution 1: Update requirements.txt

The most common issue is package compatibility. I've already updated the requirements.txt file to use more flexible version requirements.

## Solution 2: Simplify the Application

If you still face issues, you can simplify the application by disabling certain features:

1. Open `Home.py` and add this at the top, after the imports:

```python
# Feature flags - disable heavy features if deployment issues occur
ENABLE_HEAVY_FEATURES = False  # Set to False if having deployment issues
```

2. Then, in each relevant section, add conditional checks:

```python
if ENABLE_HEAVY_FEATURES:
    # Code that uses heavy dependencies
else:
    # Simplified version or information message
```

## Solution 3: Use packages.txt for System Dependencies

If your app requires system-level dependencies, create a file called `packages.txt` in the root of your repository with the required packages. For example:

```
libblas-dev
liblapack-dev
```

## Solution 4: Use a requirements-minimal.txt

Create a minimal version of requirements.txt for initial deployment:

```
streamlit
pandas
numpy
plotly
requests
```

Once the app is deployed and working, you can gradually add more dependencies.

## Solution 5: Check Dependency Conflicts

Some packages may conflict with each other. Try commenting out potentially conflicting packages in your requirements.txt file and gradually add them back one by one.

## Solution 6: Check Streamlit Cloud Logs

In the Streamlit Cloud dashboard, check the detailed logs for your app to see exactly what's failing during installation.

## Solution 7: Use Python Version Specification

In the Streamlit Cloud advanced settings, you can specify a Python version. Try setting it to Python 3.9 or 3.10 which have good compatibility with most packages.

## Contact Streamlit Support

If you continue to face issues, Streamlit has a community forum at https://discuss.streamlit.io/ where you can get help from other users and the Streamlit team.