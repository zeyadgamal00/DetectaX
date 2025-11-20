import base64
import io
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import os
import time
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime

try:
    from navbar.navbar import render_navbar
    from footer.footer import render_footer
except ImportError:
    st.error("Error: Could not import navbar or footer components.")


    def render_navbar():
        pass


    def render_footer():
        pass

try:
    from api_client import classify_image

    try:
        from utils.helpers import CLASS_NAMES
    except ImportError:
        CLASS_NAMES = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
except ImportError:
    def classify_image(image, top_k=5):
        time.sleep(1.0)
        return {"class": "Demo Class", "confidence": 0.85, "method": "Simulated"}


    CLASS_NAMES = ["Demo Class", "Other A", "Other B", "Other C", "Other D"]

ICON_BRAIN = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-cpu"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>"""
ICON_UPLOAD = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-upload-cloud"><polyline points="16 16 12 12 8 16"></polyline><line x1="12" y1="12" x2="12" y2="21"></line><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path><polyline points="16 16 12 12 8 16"></polyline></svg>"""
ICON_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-settings"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>"""
ICON_RESULTS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-activity"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>"""
ICON_RESULTS_PLACEHOLDER = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" class="feather feather-activity"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>"""
ICON_ARROW_DOWN = """<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-arrow-down-circle"><circle cx="12" cy="12" r="10"></circle><polyline points="8 12 12 16 16 12"></polyline><line x1="12" y1="8" x2="12" y2="16"></line></svg>"""
ICON_STATS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-pie-chart"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path><path d="M22 12A10 10 0 0 0 12 2v10z"></path></svg>"""
ICON_LIST = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-list"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></svg>"""
ICON_IMAGE = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>"""

ICON_TAGS = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" fill="currentColor" class="bi bi-tags-fill" viewBox="0 0 16 16">
  <path d="M2 2a1 1 0 0 1 1-1h4.586a1 1 0 0 1 .707.293l7 7a1 1 0 0 1 0 1.414l-4.586 4.586a1 1 0 0 1-1.414 0l-7-7A1 1 0 0 1 2 6.586V2zm3.5 4a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3z"/>
  <path d="M1.293 7.793A1 1 0 0 1 1 7.086V2a1 1 0 0 0-1 1v4.586a1 1 0 0 0 .293.707l7 7a1 1 0 0 0 1.414 0l.043-.043-7.457-7.457z"/>
</svg>"""

clf_icon_path = "clf_icon_blue.svg"

with open(clf_icon_path, "w") as f:
    f.write(ICON_TAGS.replace("currentColor", "#00CCFF"))

st.set_page_config(
    page_title="Image Classification - DEPI",
    page_icon=clf_icon_path,
    layout="wide"
)


def load_css(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(BASE_DIR, "..", "assets", "global.css")
load_css(assets_path)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    @import url("https.cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css");

    html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }

    .body-bg {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -2;
        background: linear-gradient(-45deg, #020c1a, #0b2f4f, #005f73, #0a9396);
        background-size: 400% 400%; animation: gradientBG 20s ease infinite;
    }
    @keyframes gradientBG { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    [data-testid="stAppViewContainer"] { background: transparent; color: #FFFFFF; }
    [data-testid="stSidebar"] > div:first-child { background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(8px); border-right: 1px solid rgba(255, 255, 255, 0.1); }

    body, html, button, a, div, span, input { 
        cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="%2300CCFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path><circle cx="7" cy="7" r="2"></circle></svg>') 16 16, auto !important; 
    }

    .custom-loader { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; padding-top: 150px; }
    .custom-loader .loader-spinner { width: 50px; height: 50px; border: 4px solid rgba(255, 255, 255, 0.2); border-top-color: #00CCFF; border-radius: 50%; animation: spin 1s linear infinite; }
    .custom-loader p { font-weight: 600; color: #E0E0E0; animation: pulse-text 1.5s infinite ease-in-out; margin: 10px 0 0 0; }
    @keyframes pulse-text { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    .custom-toast-container { position: fixed; top: 80px; right: 20px; z-index: 99999; width: 350px; animation: toastLifecycle 5s forwards; }
    @keyframes toastLifecycle { 0% { opacity: 0; transform: translateX(100%); } 10% { opacity: 1; transform: translateX(0); } 80% { opacity: 1; transform: translateX(0); } 100% { opacity: 0; transform: translateX(100%); pointer-events: none; } }
    .toast-box { background: rgba(2, 12, 26, 0.95); backdrop-filter: blur(10px); border-left: 5px solid; border-radius: 4px; padding: 15px 20px; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5); color: #fff; margin-bottom: 10px; display: flex; align-items: center; gap: 15px; }
    .toast-warning { border-color: #FFCC00; } .toast-warning .icon { color: #FFCC00; font-size: 1.5rem; }
    .toast-error { border-color: #FF4136; } .toast-error .icon { color: #FF4136; font-size: 1.5rem; }
    .toast-success { border-color: #00CCFF; } .toast-success .icon { color: #00CCFF; font-size: 1.5rem; }
    .toast-content h4 { margin: 0 0 5px 0; font-size: 1rem; font-weight: 700; color: #fff; }
    .toast-content p { margin: 0; font-size: 0.85rem; color: #ddd; }

    .main-header-container { margin-top: -80px !important; margin-bottom: 40px !important; }

   .main-header-container { display: flex; align-items: center; gap: 20px; margin-bottom: 1rem; }
    .main-header-container .icon-box { display: flex; justify-content: center; align-items: center; color: #00CCFF; text-shadow: 0 0 15px rgba(0, 204, 255, 0.5); }
    .main-header-container .text-box h1 { font-size: 2.75rem; font-weight: 700; margin: 0; line-height: 1.1; background: linear-gradient(90deg, #33DFFF, #00CCFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .main-header-container .text-box p { font-size: 1.1rem; font-weight: 300; color: #BBBBBB; margin: 0.5rem 0 0 0; }
    .styled-hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 204, 255, 0), rgba(0, 204, 255, 0.5), rgba(0, 204, 255, 0)); margin-top: 0.5rem; margin-bottom: 1.5rem; }

    .section-header { border-bottom: 2px solid #00CCFF; padding-bottom: 10px; margin-bottom: 1.5rem; font-weight: 600; font-size: 1.5rem; display: flex; align-items: center; gap: 12px; }
    .section-header svg { color: #00CCFF; }
    .stButton > button { width: 100%; background-color: #00CCFF; color: #020c1a; font-weight: 700; border: none; padding: 0.75rem; transition: all 0.3s ease; }
    .stButton > button:hover { background-color: #33DFFF; transform: translateY(-2px); }
    .stButton > button:disabled { background-color: rgba(255, 255, 255, 0.2); color: #888888; }

    .img-container { width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; padding: 20px; position: relative; border-radius: 8px; overflow: hidden; }
    .img-container img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; }

    .img-container.scan-effect { border: 2px solid rgba(0, 204, 255, 0.5); animation: pulse-border 1.5s infinite alternate; }
    .img-container.scan-effect::before { 
        content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
        background-image: linear-gradient(#00CCFF 1px, transparent 1px), linear-gradient(90deg, #00CCFF 1px, transparent 1px);
        background-size: 40px 40px; background-position: 0 0; z-index: 10; opacity: 0.3;
        animation: grid-move 3s linear infinite;
    }
    .img-container.scan-effect::after {
        content: 'ANALYZING FEATURES...';
        position: absolute; bottom: 20px; right: 20px;
        color: #00CCFF; font-family: monospace; font-weight: bold;
        background: rgba(0,0,0,0.8); padding: 5px 10px; border-radius: 4px;
        z-index: 11; border: 1px solid #00CCFF;
    }

    .img-container.completed-effect { 
        border: 2px solid #00CCFF; 
        box-shadow: 0 0 15px rgba(0, 204, 255, 0.4);
    }
    .img-container.completed-effect::after {
        content: 'Classification Complete'; 
        position: absolute; bottom: 20px; right: 20px;
        color: #00CCFF; font-family: monospace; font-weight: bold;
        background: rgba(0,0,0,0.85); 
        padding: 5px 15px 5px 35px; 
        border-radius: 4px;
        z-index: 11; 
        border: 1px solid #00CCFF;

        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%2300CCFF' class='bi bi-check-circle-fill' viewBox='0 0 16 16'%3E%3Cpath d='M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zm-3.97-3.03a.75.75 0 0 0-1.08.022L7.477 9.417 5.384 7.323a.75.75 0 0 0-1.06 1.06L6.97 11.03a.75.75 0 0 0 1.079-.02l3.992-4.99a.75.75 0 0 0-.01-1.05z'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: 10px center; 
    }

    @keyframes grid-move { 0% { background-position: 0 0; } 100% { background-position: 40px 40px; } }
    @keyframes pulse-border { from { box-shadow: 0 0 5px rgba(0, 204, 255, 0.3); } to { box-shadow: 0 0 20px rgba(0, 204, 255, 0.6); } }

    .prob-card {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px 15px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .prob-card:hover {
        border-color: #00CCFF;
        transform: translateX(5px);
        background: rgba(0, 204, 255, 0.08);
        box-shadow: -5px 0 15px rgba(0, 204, 255, 0.1);
    }

    .prob-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .prob-class {
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .prob-rank {
        color: #00CCFF;
        font-family: monospace;
        font-size: 0.9rem;
        opacity: 0.7;
        background: rgba(0, 204, 255, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
    }

    .prob-score {
        color: #00CCFF;
        font-family: monospace;
        font-weight: 700;
    }

    .prob-bar-bg {
        height: 6px;
        width: 100%;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
    }

    .prob-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #005f73, #00CCFF);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 204, 255, 0.5);
        animation: loadBar 1.5s ease-out forwards;
    }

    @keyframes loadBar {
        from { width: 0; }
    }

    .browse-button-only { margin-top: 0.5rem; }
    .browse-button-only [data-testid="stFileUploader"] { border: none; background: transparent; padding: 0; }
    .browse-button-only [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] { 
        background-color: rgba(0, 204, 255, 0.05);
        border: 1px dashed rgba(0, 204, 255, 0.3);
    }
    .browse-button-only [data-testid="stFileUploader"] [data-testid="stFileUploadButton"] { width: 100%; }

    .top-prediction-card {
        background: rgba(0, 0, 0, 0.2); padding: 2rem; border-radius: 10px;
        text-align: center; border: 1px solid #00CCFF;
        box-shadow: 0 0 20px rgba(0, 204, 255, 0.15); margin-bottom: 1.5rem;
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .top-prediction-card h4 { color: #AAAAAA; font-weight: 400; margin-bottom: 0.5rem; letter-spacing: 1px; text-transform: uppercase; font-size: 0.9rem;}
    .top-prediction-card .class-name { 
        background: linear-gradient(90deg, #FFFFFF, #00CCFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 3rem; margin-bottom: 0.5rem; font-weight: 700;
    }
    .top-prediction-card .confidence-value { 
        color: #00CCFF; font-size: 2rem; margin: 0; font-weight: 600;
    }

    .confidence-list-item { margin-bottom: 12px; }
    .confidence-label { display: flex; justify-content: space-between; color: #E0E0E0; font-size: 0.9rem; margin-bottom: 4px; }
    .confidence-bg { background: rgba(255,255,255,0.1); border-radius: 5px; height: 8px; width: 100%; overflow: hidden; }
    .confidence-fill { height: 100%; background: linear-gradient(90deg, #005f73, #00CCFF); border-radius: 5px; }

    div[data-baseweb="slider"] div[role="progressbar"] { background: linear-gradient(90deg, #005f73, #00CCFF, #FFFFFF, #00CCFF, #005f73) !important; background-size: 200% 100% !important; animation: beam-flow 2.5s linear infinite !important; box-shadow: 0 0 10px rgba(0, 204, 255, 0.6); height: 8px !important; border-radius: 10px; }
    div[data-baseweb="slider"] > div > div:first-child { background: rgba(255, 255, 255, 0.25) !important; height: 8px !important; border-radius: 10px; }
    div[data-baseweb="slider"] div[role="slider"] { background-color: #020c1a !important; border: 2px solid #00CCFF !important; box-shadow: 0 0 10px rgba(0, 204, 255, 0.8) !important; width: 22px !important; height: 22px !important; }
    @keyframes beam-flow { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
</style>
""", unsafe_allow_html=True)


def show_custom_toast(message, type="warning"):
    icon = "exclamation-triangle-fill"
    title = "Notice"
    if type == "error":
        icon = "x-octagon-fill";
        title = "Error"
    elif type == "success":
        icon = "check-circle-fill";
        title = "Success"
    html_code = f"""<div class="custom-toast-container"><div class="toast-box toast-{type}"><div class="icon"><i class="bi bi-{icon}"></i></div><div class="toast-content"><h4>{title}</h4><p>{message}</p></div></div></div>"""
    st.markdown(html_code, unsafe_allow_html=True)


def generate_full_classification_report():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Classification_Report_{timestamp}.pdf"
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5 * inch)
        styles = getSampleStyleSheet()

        NEON_BLUE = colors.HexColor('#00CCFF')
        DARK_TEAL = colors.HexColor('#0a9396')
        LIGHT_ACCENT = colors.HexColor('#E0FFFF')

        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=20,
                                     textColor=NEON_BLUE, spaceAfter=16, alignment=1)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=16,
                                       textColor=DARK_TEAL, spaceAfter=8)
        normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=10, textColor=colors.black,
                                      spaceAfter=6)

        story = []
        story.append(Paragraph("Image Classification Analysis Report", title_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph("1. Input Image", heading_style))
        if st.session_state.original_image_bytes:
            original_img = Image.open(io.BytesIO(st.session_state.original_image_bytes))

            def img_to_rl(pil_img):
                buf = io.BytesIO()
                pil_img.save(buf, format='PNG')
                buf.seek(0)
                return RLImage(buf, width=4 * inch, height=3 * inch)

            story.append(img_to_rl(original_img))
            story.append(Spacer(1, 12))

        story.append(Paragraph("2. Top Prediction", heading_style))
        res = st.session_state.classification_result
        story.append(Paragraph(f"<b>Class:</b> {res['class'].title()}", normal_style))
        story.append(Paragraph(f"<b>Confidence:</b> {res['confidence']:.2%}", normal_style))
        story.append(Paragraph(f"<b>Inference Time:</b> {st.session_state.inference_time:.2f}s", normal_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph("3. Probability Breakdown", heading_style))
        if 'chart_data' in st.session_state and not st.session_state.chart_data.empty:
            df = st.session_state.chart_data
            table_data = [['Class Label', 'Probability']]
            for index, row in df.iterrows():
                table_data.append([row['Class'].title(), f"{row['Confidence']:.2%}"])

            det_table = Table(table_data, colWidths=[3 * inch, 2 * inch])
            det_table.setStyle(TableStyle([
                ('BOX', (0, 0), (-1, -1), 1.5, NEON_BLUE),
                ('BACKGROUND', (0, 0), (-1, 0), DARK_TEAL),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FFFF'), LIGHT_ACCENT]),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            story.append(det_table)
            story.append(Spacer(1, 12))

        if 'chart_data' in st.session_state:
            story.append(PageBreak())
            story.append(Paragraph("4. Visual Analytics", heading_style))
            try:
                df = st.session_state.chart_data
                custom_colors = ['#00CCFF', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03']

                fig1 = px.bar(df, x='Class', y='Confidence',
                              title="Confidence Distribution",
                              color='Class',
                              text_auto='.1%',
                              color_discrete_sequence=custom_colors)

                fig1.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'),
                                   showlegend=False)
                img_bytes1 = fig1.to_image(format="png", width=600, height=350, scale=2)
                story.append(RLImage(io.BytesIO(img_bytes1), width=6 * inch, height=3.5 * inch))
                story.append(Spacer(1, 20))

                fig2 = px.pie(df, names='Class', values='Confidence',
                              title="Probability Ratio",
                              color_discrete_sequence=custom_colors)

                fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))

                fig2.update_traces(textposition='inside', textinfo='label+percent+value')

                img_bytes2 = fig2.to_image(format="png", width=600, height=350, scale=2)
                story.append(RLImage(io.BytesIO(img_bytes2), width=6 * inch, height=3.5 * inch))

            except Exception as e:
                story.append(
                    Paragraph(f"Charts could not be generated. Ensure 'kaleido' is installed. (Error: {str(e)})",
                              normal_style))

        doc.build(story)
        return buffer.getvalue(), filename
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None, None


def file_uploader_callback():
    file = None
    if 'browse_uploader_clf' in st.session_state and st.session_state.browse_uploader_clf is not None:
        file = st.session_state.browse_uploader_clf

    if file:
        if file.file_id != st.session_state.uploaded_file_id:
            st.session_state.loading = False
            st.session_state.results_ready = False
            st.session_state.classification_result = {}
            st.session_state.chart_data = pd.DataFrame()
            st.session_state.error = None
            st.session_state.uploaded_file_id = file.file_id
            st.session_state.inference_time = 0.0
            st.session_state.original_image_bytes = file.getvalue()
    else:
        if st.session_state.uploaded_file_id is not None:
            st.session_state.loading = False
            st.session_state.results_ready = False
            st.session_state.classification_result = {}
            st.session_state.chart_data = pd.DataFrame()
            st.session_state.error = None
            st.session_state.uploaded_file_id = None
            st.session_state.inference_time = 0.0
            st.session_state.original_image_bytes = None


def main():
    st.markdown('<div class="body-bg"></div>', unsafe_allow_html=True)
    render_navbar()

    if 'loading' not in st.session_state: st.session_state.loading = False
    if 'results_ready' not in st.session_state: st.session_state.results_ready = False
    if 'classification_result' not in st.session_state: st.session_state.classification_result = {}
    if 'chart_data' not in st.session_state: st.session_state.chart_data = pd.DataFrame()
    if 'uploaded_file_id' not in st.session_state: st.session_state.uploaded_file_id = None
    if 'error' not in st.session_state: st.session_state.error = None
    if 'inference_time' not in st.session_state: st.session_state.inference_time = 0.0
    if 'original_image_bytes' not in st.session_state: st.session_state.original_image_bytes = None

    st.markdown(f"""
        <div class="main-header-container">
            <div class="icon-box">{ICON_TAGS}</div>
            <div class="text-box">
                <h1>Image Classification</h1>
                <p>Upload an image to classify its content into predefined categories using DetectaX.</p>
            </div>
        </div>
        <hr class="styled-hr">
    """, unsafe_allow_html=True)

    col1, spacer, col2 = st.columns([1, 0.15, 1])
    FRAME_HEIGHT = 480

    with col1:
        st.markdown(f'<div class="section-header">{ICON_UPLOAD} <span>Input Image</span></div>', unsafe_allow_html=True)
        input_frame = st.container(border=True, height=FRAME_HEIGHT)
        with input_frame:
            if st.session_state.original_image_bytes:
                if st.session_state.loading:
                    scan_class = "scan-effect"
                elif st.session_state.results_ready:
                    scan_class = "completed-effect"
                else:
                    scan_class = ""

                img_b64 = base64.b64encode(st.session_state.original_image_bytes).decode()
                st.markdown(
                    f"""<div class="img-container {scan_class}"><img src="data:image/png;base64,{img_b64}" /></div>""",
                    unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="height: 445px; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 2px dashed rgba(0, 204, 255, 0.3); border-radius: 10px; background: rgba(0, 0, 0, 0.2);">
                    <div style="color: #00CCFF; margin-bottom: 15px;">{ICON_IMAGE}</div>
                    <p style="color: #AAAAAA; font-size: 1rem; margin: 0;">Waiting for image...</p>
                    <p style="color: #00CCFF; font-size: 0.9rem; margin-top: 5px;">Please use the uploader below</p>
                    <div style="margin-top: 15px; animation: bounce 2s infinite; color: #00CCFF;">
                        {ICON_ARROW_DOWN}
                    </div>
                </div>
                <style>
                    @keyframes bounce {{
                        0%, 20%, 50%, 80%, 100% {{transform: translateY(0);}}
                        40% {{transform: translateY(-10px);}}
                        60% {{transform: translateY(-5px);}}
                    }}
                </style>
                """, unsafe_allow_html=True)

        st.markdown('<div class="browse-button-only">', unsafe_allow_html=True)
        uploaded_file_browse = st.file_uploader(
            "Browse or Drag & Drop Image Here",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            label_visibility="visible",
            key="browse_uploader_clf",
            on_change=file_uploader_callback
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">{ICON_SETTINGS} <span>Configuration</span></div>',
                    unsafe_allow_html=True)
        top_k_slider = st.slider("Top-K Predictions (Mocked)", 1, 10, 5, 1)
        st.markdown("<br>", unsafe_allow_html=True)

        predict_btn = st.button("Classify Image", disabled=(st.session_state.original_image_bytes is None))
        if predict_btn:
            st.session_state.loading = True
            st.session_state.results_ready = False
            st.session_state.classification_result = {}
            st.session_state.error = None
            st.rerun()

    with spacer:
        status_placeholder = st.empty()
        if st.session_state.loading:
            status_placeholder.markdown(
                """<div class="status-icon-container"><div class="transfer-arrow"><span></span><span></span><span></span></div></div>""",
                unsafe_allow_html=True)
        elif st.session_state.results_ready:
            if st.session_state.error:
                status_placeholder.markdown(
                    """<div class="status-icon-container"><div class="animated-error"></div></div>""",
                    unsafe_allow_html=True)
            else:
                status_placeholder.markdown(
                    """<div class="status-icon-container"><div class="animated-checkmark"></div></div>""",
                    unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="section-header">{ICON_RESULTS} <span>Classification Results</span></div>',
                    unsafe_allow_html=True)

        if st.session_state.loading:
            with st.container(border=True, height=FRAME_HEIGHT):
                st.markdown(
                    """<div class="custom-loader">
                        <div class="loader-spinner"></div>
                        <p>Analyzing Image via Azure...</p>
                        <small>This may take a moment.</small>
                    </div>""",
                    unsafe_allow_html=True)

        elif st.session_state.results_ready:
            if st.session_state.error:
                show_custom_toast(f"Error: {st.session_state.error}", "error")
            elif st.session_state.classification_result:
                show_custom_toast("Classification Completed Successfully!", "success")

                prediction = st.session_state.classification_result
                st.markdown(f"""
                <div class="top-prediction-card">
                    <h4>Identified Class</h4>
                    <h2 class="class-name">{prediction['class'].title()}</h2>
                    <h1 class="confidence-value">{prediction['confidence']:.1%}</h1>
                    <p style="color: #AAAAAA; font-size: 0.8rem; margin-top: 0.5rem;">Confidence Score</p>
                </div>
                """, unsafe_allow_html=True)

                if not st.session_state.chart_data.empty:
                    st.markdown(f"""
                                    <div style="display: flex; align-items: center; gap: 10px; margin-top: 20px; margin-bottom: 15px;">
                                        <div style="color: #00CCFF;">{ICON_LIST}</div>
                                        <h3 style="margin: 0; color: #E0E0E0; font-size: 1.2rem;">Probability Breakdown</h3>
                                    </div>
                                    """, unsafe_allow_html=True)

                    for idx, row in st.session_state.chart_data.iterrows():
                        rank = idx + 1
                        width_pct = row['Confidence'] * 100
                        class_name = row['Class'].title()
                        confidence_txt = f"{row['Confidence']:.1%}"

                        opacity = 0.3 + (row['Confidence'] * 0.7)

                        st.markdown(f"""
                                         <div class="prob-card">
                                             <div class="prob-header">
                                                 <div class="prob-class">
                                                    <span class="prob-rank">#{rank}</span>
                                                    {class_name}
                                                 </div>
                                                 <div class="prob-score">{confidence_txt}</div>
                                             </div>
                                             <div class="prob-bar-bg">
                                                 <div class="prob-bar-fill" style="width: {width_pct}%; opacity: {opacity};"></div>
                                             </div>
                                         </div>
                                         """, unsafe_allow_html=True)

            else:
                st.error("Unknown error.")

        else:
            with st.container(border=True, height=FRAME_HEIGHT):
                st.markdown(f"""
                    <div style="height: 100%; min-height: 400px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; color: #777;">
                        <div style="margin-bottom: 5px; opacity: 0.6;">{ICON_RESULTS_PLACEHOLDER.replace('width="64"', 'width="64"')}</div>
                        <p style="margin: 0; margin-bottom: 10px; font-size: 1.1rem; font-weight: 500;">Results will appear here</p>
                        <small style="opacity: 0.8;margin-bottom: 60px;">Upload an image and click Classify.</small>
                    </div>
                    """, unsafe_allow_html=True)

    if st.session_state.loading:
        try:
            if not st.session_state.original_image_bytes:
                st.session_state.error = "No image found."
                st.session_state.loading = False
                st.session_state.results_ready = True
                st.rerun()
                return

            img_for_processing = Image.open(io.BytesIO(st.session_state.original_image_bytes))
            start_time = time.time()
            result = classify_image(img_for_processing)
            end_time = time.time()
            st.session_state.inference_time = end_time - start_time
            st.session_state.classification_result = result

            top_conf = result['confidence']
            df_data = [{'Class': result['class'], 'Confidence': top_conf}]
            remaining = 1.0 - top_conf
            other_classes = [c for c in CLASS_NAMES if c != result['class']]

            for _ in range(top_k_slider - 1):
                if not other_classes: break
                c = np.random.choice(other_classes)
                other_classes.remove(c)
                val = np.random.uniform(0, remaining)
                remaining -= val
                df_data.append({'Class': c, 'Confidence': val})

            st.session_state.chart_data = pd.DataFrame(df_data)

        except Exception as e:
            st.session_state.error = str(e)

        st.session_state.loading = False
        st.session_state.results_ready = True
        st.rerun()
    if st.session_state.results_ready and not st.session_state.error and not st.session_state.chart_data.empty:
        st.markdown("---")
        st.markdown(f"""
            <div class="section-header" style="border-color: #00CCFF; margin-top: 1.5rem;"> 
                {ICON_STATS}
                <span>Classification Statistics</span>
            </div>
        """, unsafe_allow_html=True)

        df = st.session_state.chart_data

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig_bar = px.bar(df, x='Class', y='Confidence', title="Confidence Distribution",
                             color='Confidence', color_continuous_scale='Blues',
                             text_auto='.1%')

            fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_chart2:
            fig_pie = px.pie(df, names='Class', values='Confidence', title="Probability Ratio",
                             color_discrete_sequence=px.colors.sequential.Teal)

            fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')

            fig_pie.update_traces(textposition='inside', textinfo='label+percent+value')

            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        st.markdown("""
                <style>
                    [data-testid="stDataFrame"] {
                        background: rgba(0, 0, 0, 0.2);
                        border: 1px solid rgba(0, 204, 255, 0.3);
                        border-radius: 10px;
                    }
                    [data-testid="stDataFrame"] th {
                        background-color: rgba(0, 204, 255, 0.15) !important;
                        border-bottom: 2px solid #00CCFF !important;
                        color: white !important;
                    }
                </style>
                """, unsafe_allow_html=True)

        st.markdown("### Detailed Probabilities")
        display_df = df.copy()
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2%}")
        st.dataframe(display_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### PDF Report")
        if st.button("Generate Full PDF Report", use_container_width=True, key="pdf_report_btn_main"):
            with st.spinner("Generating PDF report (Includes Images, Tables & Charts)..."):
                pdf_data, filename = generate_full_classification_report()
                if pdf_data:
                    show_custom_toast("PDF Generated Successfully!", "success")
                    st.download_button(label="Download PDF Report", data=pdf_data, file_name=filename,
                                       mime="application/pdf", use_container_width=True)
                else:
                    show_custom_toast("Failed to generate PDF.", "error")
    st.markdown("---")
    st.markdown("""
    <style>
        .summary-metric-card { background-color: rgba(0, 0, 0, 0.2); border: 1px solid rgba(0, 204, 255, 0.3); border-radius: 10px; padding: 20px 10px; text-align: center; transition: all 0.3s ease-in-out; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; }
        .summary-metric-card:hover { border-color: #00CCFF; box-shadow: 0 0 15px rgba(0, 204, 255, 0.3); transform: translateY(-5px); background-color: rgba(0, 204, 255, 0.05); }
        .summary-metric-card .label { color: #AAAAAA; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
        .summary-metric-card .value { color: #FFFFFF; font-size: 2rem; font-weight: 700; text-shadow: 0 0 10px rgba(0, 204, 255, 0.5); line-height: 1.2; }
        .summary-metric-card .sub-text { font-size: 0.85rem; margin-top: 5px; font-weight: 500; }
        .positive { color: #00CCFF; } .negative { color: #FF4136; } .neutral { color: #777; }
    </style>
    """, unsafe_allow_html=True)

    metric_time_val, metric_time_txt, metric_time_cls = "N/A", "Ready", "neutral"
    metric_top1_val, metric_top1_txt, metric_top1_cls = "N/A", "No Data", "neutral"
    metric_status_val, metric_status_txt, metric_status_cls = "Idle", "Waiting", "neutral"

    if st.session_state.results_ready and not st.session_state.error:
        inf_time = st.session_state.inference_time
        metric_time_val = f"{inf_time:.2f}s"
        if inf_time < 1.0:
            metric_time_txt = "Fast";
            metric_time_cls = "positive"
        else:
            metric_time_txt = "Normal";
            metric_time_cls = "negative"

        if st.session_state.classification_result:
            top_conf = st.session_state.classification_result['confidence']
            metric_top1_val = f"{top_conf:.1%}"
            if top_conf >= 0.8:
                metric_top1_txt = "High Confidence";
                metric_top1_cls = "positive"
            else:
                metric_top1_txt = "Low Confidence";
                metric_top1_cls = "negative"
            metric_status_val = "Done";
            metric_status_txt = "Success";
            metric_status_cls = "positive"

    colB, colC, colD = st.columns(3)
    with colB:
        st.markdown(
            f"""<div class="summary-metric-card"><div class="label">Inference Time</div><div class="value">{metric_time_val}</div><div class="sub-text {metric_time_cls}">{metric_time_txt}</div></div>""",
            unsafe_allow_html=True)
    with colC:
        st.markdown(
            f"""<div class="summary-metric-card"><div class="label">Top-1 Accuracy</div><div class="value">{metric_top1_val}</div><div class="sub-text {metric_top1_cls}">{metric_top1_txt}</div></div>""",
            unsafe_allow_html=True)
    with colD:
        st.markdown(
            f"""<div class="summary-metric-card"><div class="label">Model Status</div><div class="value">{metric_status_val}</div><div class="sub-text {metric_status_cls}">{metric_status_txt}</div></div>""",
            unsafe_allow_html=True)

    render_footer()


if __name__ == "__main__":
    main()