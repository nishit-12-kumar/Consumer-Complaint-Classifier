import streamlit as st

def render_confidence_bars(confidence_dict):
    """
    Renders Streamlit progress bars for model prediction confidences.
    """
    st.markdown("### Confidence Distribution")
    
    # Sort the dictionary by highest confidence first
    sorted_conf = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
    
    for label, score in sorted_conf:
        # Convert decimal to percentage
        percentage = score * 100
        
        # Display the label and percentage text
        st.write(f"**{label}**: {percentage:.2f}%")
        
        # Display the progress bar (Streamlit expects a value between 0.0 and 1.0)
        # We cap it at 1.0 just in case of floating point rounding errors
        st.progress(min(score, 1.0))