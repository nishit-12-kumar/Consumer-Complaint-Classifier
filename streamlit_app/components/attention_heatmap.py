import streamlit as st

def render_attention_heatmap(word_importances):
    """
    Renders an HTML block where words are highlighted based on their importance score.
    Higher scores get a darker/more opaque background color.
    """
    st.markdown("### Explainable AI: Token Attention")
    st.caption("Words highlighted in darker red had the highest influence on the model's prediction.")
    
    # CSS UPDATE: Changed background to #ffffff (white), added a light border, 
    # and added white-space/word-wrap rules to force text to stay inside the box.
    html_content = (
        "<div style='"
        "line-height: 2.2; font-size: 16px; padding: 15px; border-radius: 5px; "
        "background-color: #ffffff; color: #000; border: 1px solid #e0e0e0; "
        "white-space: normal; word-wrap: break-word; overflow-wrap: break-word;"
        "'>"
    )
    
    for word, score in word_importances:
        alpha = min(score, 0.8)
        bg_color = f"rgba(255, 99, 71, {alpha})"
        
        # CSS UPDATE: Added 'display: inline-block' and 'margin-bottom' so that 
        # when the words drop to the next line, the spacing stays perfectly neat.
        html_content += (
            f"<span style='background-color: {bg_color}; padding: 2px 4px; "
            f"border-radius: 4px; margin-right: 4px; margin-bottom: 4px; "
            f"display: inline-block;'>{word}</span>"
        )
        
    html_content += "</div>"
    
    st.markdown(html_content, unsafe_allow_html=True)