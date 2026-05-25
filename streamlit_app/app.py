# import streamlit as st

# # Configure the page settings (must be the first Streamlit command)
# st.set_page_config(
#     page_title="Complaint Classifier",
#     page_icon="🏦",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

import streamlit as st
from components.ui_tweaks import apply_custom_css # NEW IMPORT

st.set_page_config(
    page_title="Complaint Classifier",
    page_icon="🏦",
    layout="wide")
apply_custom_css() # APPLY CSS


def main():
    st.title("🏦 Consumer Complaints Classification System")
    st.markdown("---")
    
    st.write("""
    ### Welcome to the Complaint Classification Dashboard!
    This system automatically categorizes consumer financial complaints into predefined departments 
    using Deep Learning (BiLSTM, LSTM, CNN) and Word Embeddings.
    """)
    
    st.info("👈 **Please select a page from the sidebar to begin.**")
    
    st.markdown("""
    **Navigation Guide:**
    * **classify:** Enter raw complaint text and run the deep learning models.
    * **results:** View the classification outcome, confidence scores, and Explainable AI (XAI) insights.
    * **model_comparison:** Compare the performance of different architectures and download the evaluation report.
    """)

if __name__ == "__main__":
    main()