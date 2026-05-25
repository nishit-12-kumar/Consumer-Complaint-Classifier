import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
    /* 1. Beautiful smooth buttons */
    div.stButton > button:first-child {
        background-color: #0052cc;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #003d99;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,82,204,0.3);
    }
    
    /* 2. Soft borders for text input areas */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #dfe1e6;
        padding: 12px;
        background-color: #fafbfc;
    }
    .stTextArea textarea:focus {
        border-color: #0052cc;
        box-shadow: 0 0 0 1px #0052cc;
    }

    /* 3. Style the Streamlit Success/Info/Warning boxes */
    div[data-testid="stAlert"] {
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)