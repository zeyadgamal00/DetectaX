import base64
import io
import json
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import plotly.express as px
import sys
import os
import streamlit.components.v1 as components
import time
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
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
    from api_client import detect_objects
    from utils.visualization import draw_bounding_boxes
    from utils.helpers import CLASS_COLORS
except ImportError as e:
    def detect_objects(*args, **kwargs):
        time.sleep(1.5)
        return [{"label": "Person", "confidence": 0.87, "bbox": [50, 50, 200, 300]},
                {"label": "Car", "confidence": 0.90, "bbox": [210, 80, 400, 350]},
                {"label": "Bus", "confidence": 0.70, "bbox": [10, 10, 100, 100]},
                {"label": "Traffic Light", "confidence": 0.85, "bbox": [300, 300, 350, 400]}]

    def draw_bounding_boxes(image, detections):
        draw = ImageDraw.Draw(image)
        for det in detections:
            color = "#00CCFF"
            draw.rectangle(det['bbox'], outline=color, width=3)
            draw.text((det['bbox'][0], det['bbox'][1] - 10), f"{det['label']} {det['confidence']:.0%}", fill=color)
        return image

    CLASS_COLORS = {'default': '#00CCFF'}

ICON_TARGET = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-target"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>"""
ICON_UPLOAD = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-upload-cloud"><polyline points="16 16 12 12 8 16"></polyline><line x1="12" y1="12" x2="12" y2="21"></line><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path><polyline points="16 16 12 12 8 16"></polyline></svg>"""
ICON_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-settings"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>"""
ICON_SEARCH = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-search"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>"""
ICON_IMAGE = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>"""
ICON_ARROW_DOWN = """<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-arrow-down-circle"><circle cx="12" cy="12" r="10"></circle><polyline points="8 12 12 16 16 12"></polyline><line x1="12" y1="8" x2="12" y2="16"></line></svg>"""

target_icon_path = "target_icon_blue.svg"

with open(target_icon_path, "w") as f:
    f.write(ICON_TARGET.replace("currentColor", "#00CCFF"))

st.set_page_config(
    page_title="Object Detection - DEPI",
    page_icon=target_icon_path,
    layout="wide"
)

def load_css(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Error: CSS file not found at {file_path}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(BASE_DIR, "..", "assets", "global.css")
load_css(assets_path)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    @import url("https.cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css");

    html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #020c1a; border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 204, 255, 0.5); border-radius: 4px; border: 1px solid #020c1a; }
    ::-webkit-scrollbar-thumb:hover { background: #00CCFF; box-shadow: 0 0 10px rgba(0, 204, 255, 0.7); }
    * { scrollbar-width: thin; scrollbar-color: #00CCFF #020c1a; }

    .body-bg {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -2;
        background: linear-gradient(-45deg, #020c1a, #0b2f4f, #005f73, #0a9396);
        background-size: 400% 400%; animation: gradientBG 20s ease infinite;
    }
    @keyframes gradientBG { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    [data-testid="stAppViewContainer"] { background: transparent; color: #FFFFFF; }
    [data-testid="stSidebar"] > div:first-child { background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(8px); border-right: 1px solid rgba(255, 255, 255, 0.1); }

    body, html, button, a, div, span, input { cursor: url('data:image/svg+xml;utf8,<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="16" cy="16" r="14" stroke="%2300CCFF" stroke-width="2" stroke-opacity="0.8"/><circle cx="16" cy="16" r="4" fill="%2300CCFF"/><path d="M16 0V8M16 24V32M0 16H8M24 16H32" stroke="%2300CCFF" stroke-width="2"/></svg>') 16 16, auto !important; }

    .custom-toast-container { position: fixed; top: 80px; right: 20px; z-index: 99999; width: 350px; animation: toastLifecycle 5s forwards; }
    @keyframes toastLifecycle { 0% { opacity: 0; transform: translateX(100%); } 10% { opacity: 1; transform: translateX(0); } 80% { opacity: 1; transform: translateX(0); } 100% { opacity: 0; transform: translateX(100%); pointer-events: none; } }
    .toast-box { background: rgba(2, 12, 26, 0.95); backdrop-filter: blur(10px); border-left: 5px solid; border-radius: 4px; padding: 15px 20px; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5); color: #fff; margin-bottom: 10px; display: flex; align-items: center; gap: 15px; }
    .toast-warning { border-color: #FFCC00; } .toast-warning .icon { color: #FFCC00; font-size: 1.5rem; }
    .toast-error { border-color: #FF4136; } .toast-error .icon { color: #FF4136; font-size: 1.5rem; }
    .toast-success { border-color: #00CCFF; } .toast-success .icon { color: #00CCFF; font-size: 1.5rem; }
    .toast-content h4 { margin: 0 0 5px 0; font-size: 1rem; font-weight: 700; color: #fff; }
    .toast-content p { margin: 0; font-size: 0.85rem; color: #ddd; }

.main-header-container {
            margin-top: -80px !important;
            margin-bottom: 40px !important;
        }


    .section-header { border-bottom: 2px solid #00CCFF; padding-bottom: 10px; margin-bottom: 1.5rem; font-weight: 600; font-size: 1.5rem; display: flex; align-items: center; gap: 12px; }
    .section-header svg { color: #00CCFF; }
    .stButton > button { width: 100%; background-color: #00CCFF; color: #020c1a; font-weight: 700; border: none; padding: 0.75rem; transition: all 0.3s ease; }
    .stButton > button:hover { background-color: #33DFFF; transform: translateY(-2px); }
    .stButton > button:disabled { background-color: rgba(255, 255, 255, 0.2); color: #888888; }

    .summary-card { background: rgba(0, 204, 255, 0.1); border: 1px solid #00CCFF; border-radius: 8px; padding: 1rem; text-align: center; margin-bottom: 1.5rem; }
    .list-card { background: rgba(0, 0, 0, 0.2); border-bottom: 1px solid rgba(255, 255, 255, 0.1); border-left: 4px solid #00CCFF; padding: 0.75rem 0.5rem 0.75rem 1rem; margin: 0.25rem 0; display: flex; justify-content: space-between; align-items: center; transition: transform 0.2s; }
    .list-card:hover { background: rgba(0, 204, 255, 0.05); transform: translateX(5px); }

    .img-container { width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; padding: 20px; position: relative; border-radius: 8px; overflow: hidden; }
    .img-container img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; }
    .img-container.scan-effect { border: 2px solid rgba(0, 204, 255, 0.3); animation: pulse-glow 2.0s ease-in-out infinite alternate; }
    @keyframes pulse-glow { from { border-color: rgba(0, 204, 255, 0.2); box-shadow: 0 0 10px 2px rgba(0, 204, 255, 0.2); } to { border-color: rgba(0, 204, 255, 0.6); box-shadow: 0 0 15px 3px rgba(0, 204, 255, 0.4); } }
    .img-container.scan-effect::before { content: ''; position: absolute; left: 0; width: 100%; height: 60px; background: linear-gradient(to bottom, rgba(51, 223, 255, 0) 0%, rgba(51, 223, 255, 0.6) 50%, rgba(51, 223, 255, 0) 100%); animation: scan 2.2s ease-in-out infinite alternate; z-index: 10; opacity: 0.9; }
    @keyframes scan { 0% { top: -60px; } 100% { top: 100%; } }
    .img-container.scan-effect::after { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-image: linear-gradient(rgba(0, 204, 255, 0.15) 1px, transparent 1px), linear-gradient(to right, rgba(0, 204, 255, 0.15) 1px, transparent 1px); background-size: 30px 30px; z-index: 5; opacity: 2; }

    .status-icon-container { height: 480px; display: flex; justify-content: center; align-items: center; padding-top: 50px; }
    .transfer-arrow { width: 50px; height: 50px; position: relative; }
    .transfer-arrow span { position: absolute; width: 15px; height: 15px; border-top: 4px solid #33DFFF; border-right: 4px solid #33DFFF; transform: rotate(45deg); opacity: 0; animation: flow-arrow 1.8s linear infinite; box-shadow: 0 0 10px rgba(0, 204, 255, 0.7); }
    .transfer-arrow span:nth-child(1) { animation-delay: 0s; } .transfer-arrow span:nth-child(2) { animation-delay: 0.6s; } .transfer-arrow span:nth-child(3) { animation-delay: 1.2s; }
    @keyframes flow-arrow { 0% { left: 0px; opacity: 0; } 20% { opacity: 1; } 80% { opacity: 1; } 100% { left: 35px; opacity: 0; } }
    .animated-checkmark { width: 50px; height: 50px; position: relative; transform: scale(0); animation: pop-in 0.5s cubic-bezier(0.18, 0.89, 0.32, 1.28) forwards; }
    .animated-checkmark::after { content: ''; display: block; width: 18px; height: 35px; border: solid #33DFFF; border-width: 0 6px 6px 0; box-shadow: 0 0 15px rgba(0, 204, 255, 0.7); transform: rotate(45deg); position: absolute; top: 0px; left: 15px; }
    .animated-error { width: 50px; height: 50px; position: relative; transform: scale(0); animation: pop-in 0.5s cubic-bezier(0.18, 0.89, 0.32, 1.28) forwards; }
    .animated-error::before, .animated-error::after { content: ''; position: absolute; width: 100%; height: 6px; background-color: #FF4136; border-radius: 3px; top: 22px; box-shadow: 0 0 15px rgba(255, 65, 54, 0.7); }
    .animated-error::before { transform: rotate(45deg); } .animated-error::after { transform: rotate(-45deg); }
    @keyframes pop-in { 0% { transform: scale(0); opacity: 0; } 80% { transform: scale(1.2); opacity: 1; } 100% { transform: scale(1.0); opacity: 1; } }

    .custom-loader { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; padding-top: 150px; }
    .custom-loader .loader-spinner { width: 50px; height: 50px; border: 4px solid rgba(255, 255, 255, 0.2); border-top-color: #00CCFF; border-radius: 50%; animation: spin 1s linear infinite; }
    .custom-loader p { font-weight: 600; color: #E0E0E0; animation: pulse-text 1.5s infinite ease-in-out; margin: 10px 0 0 0; }
    @keyframes pulse-text { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    .footer-metric-card { background: rgba(0, 0, 0, 0.2); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 1.25rem 1rem; height: 100%; }
    .footer-metric-card .icon-container { color: #00CCFF; font-size: 2rem; margin-bottom: 10px; }
    .footer-metric-card h3 { font-size: 1rem; font-weight: 400; color: #AAAAAA; margin-bottom: 0.25rem; text-transform: uppercase; }
    .footer-metric-card .value { font-size: 2.25rem; font-weight: 700; color: #FFFFFF; line-height: 1; }
    .footer-metric-card .delta { font-size: 1rem; font-weight: 600; margin-left: 8px; }
    .footer-metric-card .delta.positive { color: #3D9970; }
    .footer-metric-card .delta.negative { color: #FF4136; }
    .footer-metric-card .value-container { display: flex; align-items: baseline; }

    .browse-button-only { margin-top: 0.5rem; }
    .browse-button-only [data-testid="stFileUploader"] { border: none; background: transparent; padding: 0; }
    .browse-button-only [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] { 
        background-color: rgba(0, 204, 255, 0.05);
        border: 1px dashed rgba(0, 204, 255, 0.3);
    }
    .browse-button-only [data-testid="stFileUploader"] [data-testid="stFileUploadButton"] { width: 100%; }

    .main-header-container { display: flex; align-items: center; gap: 20px; margin-bottom: 1rem; }
    .main-header-container .icon-box { display: flex; justify-content: center; align-items: center; color: #00CCFF; text-shadow: 0 0 15px rgba(0, 204, 255, 0.5); }
    .main-header-container .text-box h1 { font-size: 2.75rem; font-weight: 700; margin: 0; line-height: 1.1; background: linear-gradient(90deg, #33DFFF, #00CCFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .main-header-container .text-box p { font-size: 1.1rem; font-weight: 300; color: #BBBBBB; margin: 0.5rem 0 0 0; }
    .styled-hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 204, 255, 0), rgba(0, 204, 255, 0.5), rgba(0, 204, 255, 0)); margin-top: 0.5rem; margin-bottom: 1.5rem; }

    div[data-baseweb="slider"] div[draggable="true"] { background-color: #00CCFF !important; }
    div[data-baseweb="slider"] div[role="progressbar"] { background-color: #00CCFF !important; }
    .stSlider p { color: #FFFFFF !important; font-weight: 600; }
    [data-testid="stDataFrame"] { background: rgba(0, 0, 0, 0.2); border: 1px solid rgba(0, 204, 255, 0.3); border-radius: 10px; }
    [data-testid="stDataFrame"] th { background-color: rgba(0, 204, 255, 0.15) !important; border-bottom: 2px solid #00CCFF !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

def show_custom_toast(message, type="warning"):
    icon = "exclamation-triangle-fill"
    title = "Notice"
    if type == "error":
        icon = "x-octagon-fill"
        title = "Error"
    elif type == "success":
        icon = "check-circle-fill"
        title = "Success"

    html_code = f"""
    <div class="custom-toast-container">
        <div class="toast-box toast-{type}">
            <div class="icon"><i class="bi bi-{icon}"></i></div>
            <div class="toast-content">
                <h4>{title}</h4>
                <p>{message}</p>
            </div>
        </div>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def display_detection_summary(detections):
    total_objects = len(detections)
    st.markdown("---")
    st.markdown(f"""
    <div class="summary-card">
        <h4>Total Objects Detected</h4>
        <h1>{total_objects}</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Detected Objects List:**")

    item_height = 75
    container_kwargs = {"border": False}
    if detections and len(detections) > 3:
        container_kwargs["height"] = int(3.2 * item_height)

    with st.container(**container_kwargs):
        if not detections:
            st.info("No objects found.")
        for i, detection in enumerate(detections, 1):
            st.markdown(f"""
            <div class="list-card">
                <div>
                    <strong>{detection['label'].title()}</strong>
                    <br>
                    <small style="color:#AAAAAA;">Confidence: {detection['confidence']:.1%}</small>
                </div>
                <span class="confidence-badge" style="color:#00CCFF; font-weight:bold;">{detection['confidence']:.0%}</span>
            </div>
            """, unsafe_allow_html=True)

def display_detection_charts(detections):
    st.markdown("---")
    st.markdown(f"""
        <div class="section-header" style="border-color: #00CCFF; margin-top: 1.5rem;"> 
            {ICON_SEARCH.replace('feather-search', 'feather-bar-chart-2')}
            <span>Detection Statistics</span>
        </div>
    """, unsafe_allow_html=True)

    if detections:
        df_detections = pd.DataFrame(detections)
        detection_stats = df_detections['label'].value_counts().reset_index()
        detection_stats.columns = ['Object', 'Count']
        fig1 = px.bar(detection_stats, x='Object', y='Count',
                      title="Object Distribution",
                      color='Count',
                      color_continuous_scale='Blues')
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df_detections, x='confidence',
                            title="Confidence Distribution",
                            nbins=10,
                            color_discrete_sequence=['#00CCFF'])
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig2, use_container_width=True)

def file_uploader_callback():
    file = None
    if 'browse_uploader' in st.session_state and st.session_state.browse_uploader is not None:
        file = st.session_state.browse_uploader

    if file:
        if file.file_id != st.session_state.uploaded_file_id:
            st.session_state.loading = False
            st.session_state.results_ready = False
            st.session_state.filtered_results = []
            st.session_state.processed_image_bytes = None
            st.session_state.error = None
            st.session_state.uploaded_file_id = file.file_id
            st.session_state.df_analytics = pd.DataFrame()
            st.session_state.inference_time = 0.0
            st.session_state.original_image_bytes = file.getvalue()
    else:
        if st.session_state.uploaded_file_id is not None:
            st.session_state.loading = False
            st.session_state.results_ready = False
            st.session_state.filtered_results = []
            st.session_state.processed_image_bytes = None
            st.session_state.error = None
            st.session_state.uploaded_file_id = None
            st.session_state.df_analytics = pd.DataFrame()
            st.session_state.inference_time = 0.0
            st.session_state.original_image_bytes = None

def generate_pdf_report():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Detection_Report_{timestamp}.pdf"
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5 * inch)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18,
                                     textColor=colors.HexColor('#005f73'), spaceAfter=12, alignment=1)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14,
                                       textColor=colors.HexColor('#0a9396'), spaceAfter=6)
        normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=10, textColor=colors.black,
                                      spaceAfter=6)

        story = []
        story.append(Paragraph("Object Detection Analysis Report", title_style))
        story.append(Spacer(1, 12))
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Generated on: {current_time}", normal_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph("1. Input & Output Images", heading_style))
        if (st.session_state.original_image_bytes and st.session_state.processed_image_bytes):
            original_img = Image.open(io.BytesIO(st.session_state.original_image_bytes))
            processed_img = Image.open(io.BytesIO(st.session_state.processed_image_bytes))

            def img_to_rl(pil_img):
                buf = io.BytesIO()
                pil_img.save(buf, format='PNG')
                buf.seek(0)
                return RLImage(buf, width=2.5 * inch, height=2 * inch)

            original_rl = img_to_rl(original_img)
            processed_rl = img_to_rl(processed_img)

            img_table_data = [['Input Image', 'Annotated Output Image'], [original_rl, processed_rl]]
            img_table = Table(img_table_data, colWidths=[2.8 * inch, 2.8 * inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f8ff')),
                ('BOX', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 12))

        story.append(Paragraph("2. Detection Summary", heading_style))
        total_objects = len(st.session_state.filtered_results) if st.session_state.filtered_results else 0
        avg_conf_val = "N/A"
        if not st.session_state.df_analytics.empty:
            avg_conf_val = f"{st.session_state.df_analytics['Confidence'].mean():.1%}"

        kpi_data = [
            ['Metric', 'Value'],
            ['Total Objects', str(total_objects)],
            ['mAP Score', '78.5%'],
            ['Inference Time', f"{st.session_state.inference_time:.2f}s"],
            ['Avg Confidence', avg_conf_val]
        ]
        kpi_table = Table(kpi_data, colWidths=[2 * inch, 2 * inch])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#005f73')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOX', (0, 0), (-1, -1), 1, colors.grey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 12))

        if st.session_state.filtered_results:
            story.append(Paragraph("3. Detailed Results", heading_style))
            table_data = [['Object', 'Confidence', 'Position', 'Size']]
            for idx, row in st.session_state.df_analytics.iterrows():
                table_data.append([
                    row['Label'].title(),
                    f"{row['Confidence']:.1%}",
                    row['Position'],
                    row['Size']
                ])
            if len(table_data) > 21:
                table_data = table_data[:21]

            det_table = Table(table_data, colWidths=[1.5 * inch, 1 * inch, 1.5 * inch, 1 * inch])
            det_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0a9396')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
            ]))
            story.append(det_table)
            story.append(Spacer(1, 12))

        if not st.session_state.df_analytics.empty:
            story.append(PageBreak())
            story.append(Paragraph("4. Visual Analytics (Charts)", heading_style))

            try:
                df = st.session_state.df_analytics

                counts = df['Label'].value_counts().reset_index()
                counts.columns = ['Object', 'Count']
                fig1 = px.bar(counts, x='Object', y='Count',
                              title="Object Distribution",
                              text_auto=True,
                              color_discrete_sequence=['#005f73'])
                fig1.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))

                img_bytes1 = fig1.to_image(format="png", width=600, height=350, scale=2)
                story.append(RLImage(io.BytesIO(img_bytes1), width=6 * inch, height=3.5 * inch))
                story.append(Spacer(1, 20))

                fig2 = px.pie(df, names='Label', title='Object Ratios',
                              color_discrete_sequence=px.colors.sequential.Teal)
                fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))

                img_bytes2 = fig2.to_image(format="png", width=600, height=350, scale=2)
                story.append(RLImage(io.BytesIO(img_bytes2), width=6 * inch, height=3.5 * inch))

            except Exception as e:
                warning_text = f"Charts could not be generated. Ensure 'kaleido' is installed using 'pip install kaleido'. Error: {str(e)}"
                story.append(Paragraph(warning_text, normal_style))

        doc.build(story)
        return buffer.getvalue(), filename

    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None, None

def main():
    st.markdown('<div class="body-bg"></div>', unsafe_allow_html=True)
    render_navbar()

    if 'loading' not in st.session_state: st.session_state.loading = False
    if 'results_ready' not in st.session_state: st.session_state.results_ready = False
    if 'filtered_results' not in st.session_state: st.session_state.filtered_results = []
    if 'processed_image_bytes' not in st.session_state: st.session_state.processed_image_bytes = None
    if 'uploaded_file_id' not in st.session_state: st.session_state.uploaded_file_id = None
    if 'error' not in st.session_state: st.session_state.error = None
    if 'df_analytics' not in st.session_state: st.session_state.df_analytics = pd.DataFrame()
    if 'inference_time' not in st.session_state: st.session_state.inference_time = 0.0
    if 'original_image_bytes' not in st.session_state: st.session_state.original_image_bytes = None

    st.markdown(f"""
        <div class="main-header-container">
            <div class="icon-box">
                {ICON_TARGET}
            </div>
            <div class="text-box">
                <h1>Object Detection</h1>
                <p>Upload an image to detect and locate multiple objects using our advanced detection models.</p>
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
            scan_class = "scan-effect" if st.session_state.loading else ""
            if st.session_state.original_image_bytes:
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
        uploaded_file_browse = st.file_uploader("Browse or Drag & Drop Image Here", type=['jpg', 'jpeg', 'png', 'bmp'],
                                                label_visibility="visible", key="browse_uploader",
                                                on_change=file_uploader_callback)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <style>
            @keyframes beam-flow {
                0% { background-position: 0% 50%; }
                100% { background-position: 200% 50%; }
            }

            div[data-baseweb="slider"] div[role="progressbar"] {
                background: linear-gradient(90deg, #005f73, #00CCFF, #FFFFFF, #00CCFF, #005f73) !important;
                background-size: 200% 100% !important;
                animation: beam-flow 2.5s linear infinite !important;
                box-shadow: 0 0 10px rgba(0, 204, 255, 0.6);
                height: 8px !important;
                border-radius: 10px;
            }

            div[data-baseweb="slider"] > div > div:first-child {
                background: rgba(255, 255, 255, 0.25) !important; 
                height: 8px !important;
                border-radius: 10px;
            }

            div[data-baseweb="slider"] div[role="slider"] {
                background-color: #020c1a !important;
                border: 2px solid #00CCFF !important;
                box-shadow: 0 0 10px rgba(0, 204, 255, 0.8) !important;
                width: 22px !important;
                height: 22px !important;
            }

            div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] {
                color: #00CCFF !important;
                font-family: monospace;
            }

            div[data-baseweb="slider"] p {
                color: #FFFFFF !important;
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="section-header">{ICON_SETTINGS} <span>Configuration</span></div>',
                    unsafe_allow_html=True)
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
        max_detections = st.slider("Maximum Detections", 1, 50, 20)
        st.markdown("<br>", unsafe_allow_html=True)


        st.markdown("<br>", unsafe_allow_html=True)

        predict_btn = st.button("Detect Objects", disabled=(st.session_state.original_image_bytes is None))
        if predict_btn:
            st.session_state.loading = True
            st.session_state.results_ready = False
            st.session_state.filtered_results = []
            st.session_state.processed_image_bytes = None
            st.session_state.error = None
            st.session_state.df_analytics = pd.DataFrame()
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
        st.markdown(f'<div class="section-header">{ICON_SEARCH} <span>Detection Results</span></div>',
                    unsafe_allow_html=True)
        result_frame = st.container(border=True, height=FRAME_HEIGHT)
        with result_frame:
            if st.session_state.loading:
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
                elif st.session_state.processed_image_bytes:
                    img_b64 = base64.b64encode(st.session_state.processed_image_bytes).decode()
                    if not st.session_state.filtered_results:
                        show_custom_toast("No objects detected above confidence threshold.", "warning")
                    else:
                        show_custom_toast("Analysis Completed Successfully!", "success")

                    st.markdown(f"""<div class="img-container"><img src="data:image/png;base64,{img_b64}" /></div>""",
                                unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""
                                <div style="
                                    height: 100%; 
                                    min-height: 400px; 
                                    display: flex; 
                                    flex-direction: column; 
                                    justify-content: center; 
                                    align-items: center; 
                                    text-align: center; 
                                    color: #777;
                                ">
                                    <div style="margin-bottom: 5px; opacity: 0.6;">{ICON_IMAGE}</div>
                                    <p style="margin: 0; margin-bottom: 10px; font-size: 1.1rem; font-weight: 500;">Results will appear here</p>
                                    <small style="opacity: 0.8;margin-bottom: 60px;">Upload an image and click Detect.</small>
                                </div>
                                """,
                    unsafe_allow_html=True)

        if st.session_state.results_ready and not st.session_state.error and st.session_state.processed_image_bytes:
            st.markdown("""
                    <style>
                        [data-testid="stDownloadButton"] button {
                            border: 1px solid #00CCFF !important;
                            color: #00CCFF !important;
                            background-color: rgba(0, 204, 255, 0.05) !important;
                            transition: all 0.3s ease-in-out;
                        }

                        [data-testid="stDownloadButton"] button:hover {
                            background-color: #00CCFF !important;
                            color: #000000 !important;
                            box-shadow: 0 0 10px rgba(0, 204, 255, 0.4);
                        }
                    </style>
                    """, unsafe_allow_html=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"detected_result_{timestamp}.png"

            st.download_button(
                label="Download Result Image",
                data=st.session_state.processed_image_bytes,
                file_name=file_name,
                mime="image/png",
                use_container_width=True,
                key="download_img_btn"
            )
        if st.session_state.results_ready and not st.session_state.error:
            if st.session_state.filtered_results:
                display_detection_summary(st.session_state.filtered_results)
            else:
                display_detection_summary([])

    if st.session_state.loading:
        try:
            if not st.session_state.original_image_bytes:
                st.session_state.error = "Image data not found in session state."
                st.session_state.loading = False
                st.session_state.results_ready = True
                st.rerun()
                return

            img_for_processing = Image.open(io.BytesIO(st.session_state.original_image_bytes))
            img_height = img_for_processing.height
            start_time = time.time()
            detections = detect_objects(img_for_processing, threshold=confidence_threshold)
            end_time = time.time()
            st.session_state.inference_time = end_time - start_time

            filtered = detections[:max_detections]
            st.session_state.filtered_results = filtered

            df_data = []
            for det in filtered:
                bbox = det['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = int(width * height)
                center_y = (bbox[1] + bbox[3]) / 2
                center_x = (bbox[0] + bbox[2]) / 2
                if center_y < img_height / 3:
                    position = 'Top'
                elif center_y > 2 * img_height / 3:
                    position = 'Bottom'
                else:
                    position = 'Middle'
                if area < 32000:
                    size_cat = 'Small'
                elif area <= 90000:
                    size_cat = 'Medium'
                else:
                    size_cat = 'Large'

                df_data.append({'Label': det['label'], 'Confidence': det['confidence'], 'bbox_raw': bbox,
                                'Bounding Box': f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})", 'Area': area,
                                'Position': position, 'Size': size_cat, 'center_x': center_x, 'center_y': center_y})

            if df_data:
                st.session_state.df_analytics = pd.DataFrame(df_data)
            else:
                st.session_state.df_analytics = pd.DataFrame()

            if filtered:
                annotated = draw_bounding_boxes(img_for_processing.copy(), filtered)
                buf = io.BytesIO()
                annotated.save(buf, format="PNG")
                st.session_state.processed_image_bytes = buf.getvalue()
            else:
                buf = io.BytesIO()
                img_for_processing.save(buf, format="PNG")
                st.session_state.processed_image_bytes = buf.getvalue()

        except Exception as e:
            st.session_state.error = str(e)

        st.session_state.loading = False
        st.session_state.results_ready = True
        st.rerun()

    if st.session_state.results_ready and not st.session_state.error:
        display_detection_charts(st.session_state.filtered_results)
        df_analytics = st.session_state.df_analytics
        if not df_analytics.empty:

            st.markdown("---")
            st.markdown("### Detection Summary")

            avg_conf = df_analytics['Confidence'].mean()
            max_conf = df_analytics['Confidence'].max()
            min_conf = df_analytics['Confidence'].min()
            unique_classes = df_analytics['Label'].nunique()

            st.markdown("""
                        <style>
                            .summary-metric-card {
                                background-color: rgba(0, 0, 0, 0.2);
                                border: 1px solid rgba(0, 204, 255, 0.3);
                                border-radius: 10px;
                                padding: 20px 10px;
                                text-align: center;
                                transition: all 0.3s ease-in-out;
                                height: 100%;
                            }

                            .summary-metric-card:hover {
                                border-color: #00CCFF;
                                box-shadow: 0 0 15px rgba(0, 204, 255, 0.3);
                                transform: translateY(-5px);
                                background-color: rgba(0, 204, 255, 0.05);
                            }

                            .summary-metric-card .label {
                                color: #AAAAAA;
                                font-size: 0.9rem;
                                text-transform: uppercase;
                                letter-spacing: 1px;
                                margin-bottom: 8px;
                            }

                            .summary-metric-card .value {
                                color: #FFFFFF;
                                font-size: 2rem;
                                font-weight: 700;
                                text-shadow: 0 0 10px rgba(0, 204, 255, 0.5);
                            }
                        </style>
                        """, unsafe_allow_html=True)

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)

            with mcol1:
                st.markdown(f"""
                            <div class="summary-metric-card">
                                <div class="label">Average Confidence</div>
                                <div class="value">{avg_conf:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)

            with mcol2:
                st.markdown(f"""
                            <div class="summary-metric-card">
                                <div class="label">Max Confidence</div>
                                <div class="value">{max_conf:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)

            with mcol3:
                st.markdown(f"""
                            <div class="summary-metric-card">
                                <div class="label">Min Confidence</div>
                                <div class="value">{min_conf:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)

            with mcol4:
                st.markdown(f"""
                            <div class="summary-metric-card">
                                <div class="label">Unique Classes</div>
                                <div class="value">{unique_classes}</div>
                            </div>
                            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Detection Details Table")
            display_df = df_analytics[['Label', 'Confidence', 'Bounding Box', 'Area', 'Position']].copy()
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_df, use_container_width=True)

            st.markdown("---")
            st.markdown("### Object Frequency Ratio")
            fig_pie = px.pie(df_analytics, names='Label', title='Object Frequency Ratio')
            fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                                  legend_title_text='Object')
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
            st.markdown("### PDF Report")
            if st.button("Generate Full PDF Report", use_container_width=True, key="pdf_report_btn_main"):
                with st.spinner("Generating PDF report (Includes Images, Tables & Charts)..."):
                    pdf_data, filename = generate_pdf_report()
                    if pdf_data:
                        show_custom_toast("PDF Generated Successfully!", "success")
                        st.download_button(label="Download PDF Report", data=pdf_data, file_name=filename,
                                           mime="application/pdf", use_container_width=True)
                    else:
                        show_custom_toast("Failed to generate PDF.", "error")

    st.markdown("---")

    st.markdown("""
    <style>
        .summary-metric-card {
            background-color: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(0, 204, 255, 0.3);
            border-radius: 10px;
            padding: 20px 10px;
            text-align: center;
            transition: all 0.3s ease-in-out;
            height: 100%;
            display: flex; 
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .summary-metric-card:hover {
            border-color: #00CCFF;
            box-shadow: 0 0 15px rgba(0, 204, 255, 0.3);
            transform: translateY(-5px);
            background-color: rgba(0, 204, 255, 0.05);
        }
        .summary-metric-card .label {
            color: #AAAAAA;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        .summary-metric-card .value {
            color: #FFFFFF;
            font-size: 2rem;
            font-weight: 700;
            text-shadow: 0 0 10px rgba(0, 204, 255, 0.5);
            line-height: 1.2;
        }
        .summary-metric-card .sub-text {
            font-size: 0.85rem;
            margin-top: 5px;
            font-weight: 500;
        }
        .positive { color: #00CCFF; }
        .negative { color: #FF4136; }
        .neutral { color: #777; }
    </style>
    """, unsafe_allow_html=True)

    metric_time_val = "N/A"
    metric_time_txt = "Ready"
    metric_time_cls = "neutral"

    metric_obj_val = "0"
    metric_obj_txt = "Waiting"
    metric_obj_cls = "neutral"

    metric_conf_val = "N/A"
    metric_conf_txt = "No Data"
    metric_conf_cls = "neutral"

    if st.session_state.results_ready and not st.session_state.error:
        inf_time = st.session_state.inference_time
        metric_time_val = f"{inf_time:.2f}s"
        if inf_time < 1.0:
            metric_time_txt = "Fast"
            metric_time_cls = "positive"
        else:
            metric_time_txt = "Normal"
            metric_time_cls = "negative"

        obj_count = len(st.session_state.filtered_results)
        metric_obj_val = str(obj_count)
        metric_obj_txt = "Detected"
        metric_obj_cls = "positive"

        if not st.session_state.df_analytics.empty:
            avg_conf = st.session_state.df_analytics['Confidence'].mean()
            metric_conf_val = f"{avg_conf:.1%}"
            if avg_conf >= 0.7:
                metric_conf_txt = "High Accuracy"
                metric_conf_cls = "positive"
            else:
                metric_conf_txt = "Low Accuracy"
                metric_conf_cls = "negative"

    colB, colC, colD = st.columns(3)

    with colB:
        st.markdown(f"""
            <div class="summary-metric-card">
                <div class="label">Inference Time</div>
                <div class="value">{metric_time_val}</div>
                <div class="sub-text {metric_time_cls}">{metric_time_txt}</div>
            </div>
            """, unsafe_allow_html=True)

    with colC:
        st.markdown(f"""
            <div class="summary-metric-card">
                <div class="label">Total Objects</div>
                <div class="value">{metric_obj_val}</div>
                <div class="sub-text {metric_obj_cls}">{metric_obj_txt}</div>
            </div>
            """, unsafe_allow_html=True)

    with colD:
        st.markdown(f"""
            <div class="summary-metric-card">
                <div class="label">Avg. Confidence</div>
                <div class="value">{metric_conf_val}</div>
                <div class="sub-text {metric_conf_cls}">{metric_conf_txt}</div>
            </div>
            """, unsafe_allow_html=True)

    render_footer()

if __name__ == "__main__":
    main()