import os
import streamlit as st
from PIL import Image

from components.confidence_bar import render_confidence_bars
from components.attention_heatmap import render_attention_heatmap
from components.ui_tweaks import apply_custom_css

st.set_page_config(page_title="Classification Results", page_icon="📊", layout="wide")
apply_custom_css()

st.title("📊 Classification Results")

if 'predicted_category' not in st.session_state:
    st.info("No prediction data found. Please go to the 'Classify' page and submit a complaint first.")
else:
    # --- 1. Top Metrics Cards ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Predicted Department", value=st.session_state['predicted_category'])
    with col2:
        confidence_pct = f"{st.session_state['confidence_score'] * 100:.2f}%"
        st.metric(label="Model Confidence", value=confidence_pct)
    with col3:
        st.metric(label="Architecture", value=st.session_state['model_used'])
    
    st.markdown("---")
    
    # --- 2. Confidence Bars & Explainable AI ---
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # This renders your Confidence Distribution
        render_confidence_bars(st.session_state['confidence_dict'])
        
    with col_right:
        # This renders your highlighted text words
        render_attention_heatmap(st.session_state['word_importances'])
        
    st.markdown("---")
    
    # --- 3. Confusion Matrix Image ---
    st.markdown("### Model Confusion Matrix")
    cm_path = "artifacts/confusion_matrix.png"
    if os.path.exists(cm_path):
        image = Image.open(cm_path)
        st.image(image, caption="Confusion Matrix from Evaluation Phase", use_column_width=True)
    else:
        st.warning("Confusion matrix image not found. Ensure the model training pipeline has been run.")