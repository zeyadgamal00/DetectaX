# navbar/navbar.py
import streamlit as st
import os


def render_navbar():
    """
    Renders the fixed top navigation bar with its specific CSS.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))

    css_path = os.path.join(script_dir, "navbar.css")

    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Could not find navbar.css at {css_path}")

    logo_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-cpu"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>"""

    st.markdown(f"""
    <div class="glass-navbar">
        <div class="navbar-logo">
            <a href="/" target="_self" style="text-decoration: none; display: flex; align-items: center; color: inherit; gap: 10px;">
                {logo_svg}
                <div class="logo-text">DetectaX <span>Image Analyzer</span></div>
            </a>
        </div>
        <div class="navbar-links">
            <a href="/" target="_self">Home</a> 
            <a href="/Image_Classification" target="_self">Classification</a>
            <a href="/Object_Detection" target="_self">Detection</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
