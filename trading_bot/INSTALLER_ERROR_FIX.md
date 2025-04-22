# Fixing "Installer Returned Non-Zero Exit Code" Error

If you encounter the error message `installer returned a non-zero exit code` when deploying to Streamlit Cloud, follow these steps to resolve it:

## Quick Fix

1. In your Streamlit Cloud app settings, change the "Requirements file path" to:
   ```
   streamlit_app/requirements-minimal.txt
   ```

2. If your main file path isn't set correctly, set it to:
   ```
   streamlit_app/Home.py
   ```

3. Click "Save" and redeploy your app

## Understanding the Issue

This error typically occurs when:
- There are conflicts between package versions
- A package fails to install due to missing system dependencies
- Python version incompatibility with certain packages

## Detailed Solutions

### 1. Use the Minimal Requirements

The `requirements-minimal.txt` file contains only the essential packages with specific versions that are known to work together. This approach often resolves dependency conflicts.

### 2. Check System Dependencies

Make sure the `packages.txt` file is included in your deployment and contains:
```
libblas-dev
liblapack-dev
build-essential
```

### 3. Try a Different Python Version

In your app settings, try setting Python to version 3.9 instead of 3.10 or later.

### 4. Isolate the Problematic Package

If the error persists, try to identify which package is causing the issue:

1. Look at the deployment logs for specific error messages
2. Temporarily remove packages one by one from `requirements-minimal.txt` until deployment succeeds
3. Once identified, you can either:
   - Find a compatible version of the problematic package
   - Remove it if it's not essential

### 5. Manual Installation Approach

As a last resort, you can try:

1. Start with a minimal `requirements.txt` containing just `streamlit`
2. Deploy successfully with just that
3. Use `st.session_state` to check if other packages are already installed
4. Use `subprocess` to install missing packages at runtime (this is a workaround, not ideal)

## Still Having Issues?

If you continue to face installation problems:
- Try deploying a simpler version of the app first
- Gradually add back functionality
- Consider using a Docker-based deployment if available