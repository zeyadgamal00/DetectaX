import streamlit as st
import os


def render_footer():
    """
    Renders the fixed bottom footer with its specific CSS.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(script_dir, "footer.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Could not find footer.css at {css_path}")
    st.markdown("""
    <div class="glass-footer">
        DEPI Project Team © 2025
    </div>
    """, unsafe_allow_html=True)
