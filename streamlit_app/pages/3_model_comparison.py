import os
import json
import streamlit as st
import pandas as pd
from PIL import Image

from src.components.report_generator import ReportGenerator

# --- NEW ADDITION ---
from components.ui_tweaks import apply_custom_css

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")

# --- CALL THE CSS HERE ---
apply_custom_css()


st.title("📈 Model Performance Comparison")
st.markdown("Review the metrics for BiLSTM, LSTM, and CNN architectures.")

report_path = "artifacts/evaluation_report.json"
chart_path = "artifacts/model_comparison.png"
pdf_path = "artifacts/Model_Comparison_Report.pdf"

# 1. Display the Bar Chart
if os.path.exists(chart_path):
    st.markdown("### Accuracy & F1 Score Visualized")
    chart_img = Image.open(chart_path)
    st.image(chart_img, use_column_width=True)
else:
    st.warning("Comparison chart not found. Please run the training pipeline first.")

# 2. Display the Metrics Table
if os.path.exists(report_path):
    st.markdown("### Detailed Metrics Table")
    with open(report_path, "r") as f:
        report_data = json.load(f)
        
    df = pd.DataFrame(report_data).T
    df = df[['Accuracy', 'Precision', 'Recall', 'F1_Score']]
    
    st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'))
else:
    st.warning("Evaluation report JSON not found.")

st.markdown("---")

# 3. PDF Report Generation & Download
st.markdown("### Export Comprehensive Report")
st.write("Download a complete PDF containing your recent prediction results, Explainable AI insights, and global model metrics.")

# Package the current session state to pass to the report generator
current_session_data = {
    'complaint_text': st.session_state.get('complaint_text'),
    'predicted_category': st.session_state.get('predicted_category'),
    'confidence_dict': st.session_state.get('confidence_dict'),
    'word_importances': st.session_state.get('word_importances'),
    'model_used': st.session_state.get('model_used')
} if 'predicted_category' in st.session_state else None

if st.button("Generate & Download PDF Report"):
    with st.spinner("Compiling comprehensive PDF..."):
        generator = ReportGenerator(json_report_path=report_path, output_pdf_path=pdf_path)
        try:
            # Pass the session data to the generator
            generated_pdf = generator.generate_pdf(session_data=current_session_data)
            
            with open(generated_pdf, "rb") as pdf_file:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_file,
                    file_name="Comprehensive_Model_Report.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"Failed to generate PDF: {str(e)}")