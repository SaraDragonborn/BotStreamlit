import streamlit as st
import os
import sys

# Show a message directing users to use the Streamlit app instead
st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="📈",
    layout="wide"
)

st.title("AI Trading Bot")
st.markdown("## Trading Bot with AI-Powered Analysis")

st.info("""
This is the entry point for the AI Trading Bot. The application is now using Streamlit for its user interface.

To run the Streamlit app directly, please use one of the following commands:
```
streamlit run streamlit_app/Home.py
```

Or:
```
cd streamlit_app
streamlit run Home.py
```
""")

# Try to automatically start the app if running in this file
if __name__ == "__main__":
    st.write("Attempting to start the Streamlit app...")
    
    # Check if we're in the main directory or streamlit_app directory
    if os.path.exists("streamlit_app/Home.py"):
        # We're in the main directory
        sys.path.insert(0, "streamlit_app")
        try:
            import Home
            st.success("Successfully imported the Streamlit app!")
        except Exception as e:
            st.error(f"Error importing the Streamlit app: {str(e)}")
    else:
        st.warning("Could not find the Streamlit app. Please use the command line to run it.")
        
    st.markdown("---")
    st.markdown("© 2025 AI Trading Bot")