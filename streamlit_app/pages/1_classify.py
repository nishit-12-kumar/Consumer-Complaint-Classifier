import sys
import streamlit as st
from tensorflow.keras.models import load_model

from src.pipeline.predict_pipeline import PredictPipeline
from src.components.attention_extractor import AttentionExtractor
from src.utils import load_object
from src.logger import logging

# --- NEW ADDITION ---
from components.ui_tweaks import apply_custom_css

st.set_page_config(page_title="Classify Complaint", page_icon="📝", layout="wide")

# --- CALL THE CSS HERE ---
apply_custom_css()

st.title("📝 Enter Consumer Complaint")
st.markdown("Type or paste a financial complaint below to classify it.")

# Input Form
with st.form("classification_form"):
    complaint_text = st.text_area("Complaint Text:", height=200, placeholder="E.g., My EMI was deducted twice and my account is frozen...")
    
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Select Deep Learning Model:", ["BiLSTM", "LSTM", "CNN"])
    
    submit_button = st.form_submit_button("Classify Complaint")

if submit_button:
    if not complaint_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner(f"Analyzing text using {model_choice}..."):
            try:
                # 1. Run Prediction
                pipeline = PredictPipeline(model_name=model_choice)
                predicted_category, confidence_score, confidence_dict, predicted_class_index = pipeline.predict(complaint_text)
                
                # 2. Extract Attention for Explainable AI
                # We load the model and tokenizer here to pass them to the extractor
                tokenizer = load_object(pipeline.tokenizer_path)
                model = load_model(pipeline.model_path)
                
                extractor = AttentionExtractor(model, tokenizer, pipeline.max_len)
                word_importances = extractor.get_word_importances(complaint_text, predicted_class_index)
                
                # 3. Save to Session State
                st.session_state['complaint_text'] = complaint_text
                st.session_state['predicted_category'] = predicted_category
                st.session_state['confidence_score'] = confidence_score
                st.session_state['confidence_dict'] = confidence_dict
                st.session_state['word_importances'] = word_importances
                st.session_state['model_used'] = model_choice
                
                logging.info("Prediction and XAI extraction complete. Transitioning to results.")
                st.success("Classification Complete! Please go to the '2 Results' page to view the outcome.")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")