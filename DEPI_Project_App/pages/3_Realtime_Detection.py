import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import base64
import io
from PIL import Image, ImageFilter, ImageDraw
from datetime import datetime
from ultralytics import YOLO
import plotly.express as px

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# --- ICONS & ASSETS ---
ICON_CAMERA_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 16 16" fill="none" stroke="#00CCFF" stroke-width="1.5">
  <path d="M2 3h7a2 2 0 0 1 2 2v1l3.5-2v8l-3.5-2v1a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/>
</svg>
"""

icon_path = "live_icon.svg"
with open(icon_path, "w") as f:
    f.write(ICON_CAMERA_SVG)

st.set_page_config(
    page_title="Real-Time Detection - DetectaX",
    page_icon=icon_path,
    layout="wide",
    initial_sidebar_state="collapsed"
)

ICON_TARGET_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#00CCFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>"""
ICON_HISTORY = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-clock"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>"""
ICON_SEARCH = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-bar-chart-2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>"""
ICON_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-settings"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>"""
ICON_FILTER = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-filter"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon></svg>"""
ICON_PRIVACY = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-shield"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>"""
ICON_REC = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-video"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>"""


def render_navbar():
    st.markdown("""
        <style>
            .glass-navbar {
                position: fixed; top: 0; left: 0; width: 100%; padding: 0.8rem 3rem;
                background: rgba(10, 25, 47, 0.5); backdrop-filter: blur(12px);
                border-bottom: 1px solid rgba(255, 255, 255, 0.1); z-index: 999999;
                display: flex; align-items: center; justify-content: space-between; color: white;
            }
            .navbar-logo { display: flex; align-items: center; }
            .navbar-logo svg { width: 30px; height: 30px; margin-right: 10px; color: #00CCFF; }
            .logo-text { font-size: 1.6rem; font-weight: 700; color: #FFFFFF; line-height: 1; }
            .logo-text span { color: #00CCFF; font-weight: 400; }
            .navbar-links a { color: #DDDDDD; text-decoration: none; margin-left: 1.5rem; font-size: 1rem; font-weight: 400; transition: color 0.3s ease; position: relative; }
            .navbar-links a:hover { color: #00CCFF; }

            .badge-new {
                position: absolute; top: -12px; left: -20px; background: #00CCFF; color: #020c1a;
                font-size: 0.55rem; font-weight: 800; letter-spacing: 0.5px; line-height: 1;
                clip-path: polygon(15% 0, 100% 0, 100% 70%, 85% 100%, 0 100%, 0 30%);
                padding: 3px 8px; border-radius: 0; pointer-events: none;
                animation: pulse-tech 2s infinite ease-in-out;
            }
            @keyframes pulse-tech {
                0% { transform: scale(1); filter: drop-shadow(0 0 2px rgba(0, 204, 255, 0.6)); }
                50% { transform: scale(1.05); filter: drop-shadow(0 0 6px rgba(0, 204, 255, 1)); }
                100% { transform: scale(1); filter: drop-shadow(0 0 2px rgba(0, 204, 255, 0.6)); }
            }
            [data-testid="stHeader"] { background: transparent; z-index: 1; }
        </style>

        <div class="glass-navbar">
            <div class="navbar-logo">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-cpu"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>
                <div class="logo-text">DetectaX <span>Image Analyzer</span></div>
            </div>
            <div class="navbar-links">
                <a href="/" target="_self">Home</a> 
                <a href="/Image_Classification" target="_self">Classification</a>
                <a href="/Object_Detection" target="_self">Detection</a>
                <a href="/Realtime_Detection" target="_self" class="rtd-link" style="color: #00CCFF; font-weight: 700;">RTD <span class="badge-new">BETA</span></a>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_footer():
    st.markdown("""
        <style>
            .glass-footer {
                position: fixed; bottom: 0; left: 0; width: 100%; padding: 10px;
                background: rgba(2, 12, 26, 0.9); backdrop-filter: blur(10px);
                border-top: 1px solid rgba(0, 204, 255, 0.2); z-index: 9999;
                display: flex; justify-content: center; align-items: center;
                color: #8899A6; font-size: 0.8rem;
            }
            .glass-footer span { margin: 0 10px; }
            .glass-footer i { color: #00CCFF; }
            .main .block-container { padding-bottom: 60px; }
        </style>
        <div class="glass-footer">
            <span>DetectaX © 2025</span>
            <span>|</span>
            <span>Powered by <i class="bi bi-cpu"></i> YOLOv8</span>
            <span>|</span>
            <span><i class="bi bi-shield-lock"></i> Privacy First</span>
        </div>
    """, unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    @import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"); 

    html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }

    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #020c1a; border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 204, 255, 0.5); border-radius: 4px; border: 1px solid #020c1a; }
    ::-webkit-scrollbar-thumb:hover { background: #00CCFF; box-shadow: 0 0 10px rgba(0, 204, 255, 0.7); }
    * { scrollbar-width: thin; scrollbar-color: #00CCFF #020c1a; }

    body, html, button, a, div, span, input { 
        cursor: url('data:image/svg+xml;utf8,<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="16" cy="16" r="14" stroke="%2300CCFF" stroke-width="2" stroke-opacity="0.8"/><circle cx="16" cy="16" r="4" fill="%2300CCFF"/><path d="M16 0V8M16 24V32M0 16H8M24 16H32" stroke="%2300CCFF" stroke-width="2"/></svg>') 16 16, auto !important; 
    }

    .body-bg {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -2;
        background: linear-gradient(-45deg, #020c1a, #0b2f4f, #005f73, #0a9396);
        background-size: 400% 400%; animation: gradientBG 20s ease infinite;
    }
    @keyframes gradientBG { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

    [data-testid="stAppViewContainer"] { background: transparent; color: #FFFFFF; }

    .main-header-container { margin-top: -20px !important; margin-bottom: 40px !important; display: flex; align-items: center; gap: 20px; }
    .main-header-container .icon-box { display: flex; justify-content: center; align-items: center; color: #00CCFF; text-shadow: 0 0 15px rgba(0, 204, 255, 0.5); }
    .main-header-container .text-box h1 { font-size: 2.75rem; font-weight: 700; margin: 0; line-height: 1.1; background: linear-gradient(90deg, #33DFFF, #00CCFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .main-header-container .text-box p { font-size: 1.1rem; font-weight: 300; color: #BBBBBB; margin: 0.5rem 0 0 0; }

    .section-header { border-bottom: 2px solid #00CCFF; padding-bottom: 10px; margin-bottom: 1.5rem; font-weight: 600; font-size: 1.5rem; display: flex; align-items: center; gap: 12px; }
    .section-header svg { color: #00CCFF; }

    .video-frame { border: 2px solid rgba(0, 204, 255, 0.3); border-radius: 8px; overflow: hidden; position: relative; box-shadow: 0 0 20px rgba(0, 204, 255, 0.1); background: rgba(0, 0, 0, 0.3); }
    .video-frame img { width: 100%; display: block; }

    .summary-metric-card { background-color: rgba(0, 0, 0, 0.4); border: 1px solid rgba(0, 204, 255, 0.2); border-radius: 10px; padding: 15px 10px; text-align: center; transition: all 0.3s ease-in-out; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; }
    .summary-metric-card:hover { border-color: #00CCFF; background-color: rgba(0, 204, 255, 0.1); }
    .summary-metric-card .label { color: #AAAAAA; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .summary-metric-card .value { color: #FFFFFF; font-size: 1.5rem; font-weight: 700; text-shadow: 0 0 10px rgba(0, 204, 255, 0.5); line-height: 1.2; }
    .summary-metric-card .sub-text { font-size: 0.75rem; margin-top: 5px; font-weight: 500; }
    .positive { color: #00CCFF; } .negative { color: #FF4136; } .neutral { color: #777; }

    .history-card { background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(255, 255, 255, 0.1); border-left: 3px solid #00CCFF; border-radius: 8px; padding: 15px; margin-bottom: 10px; transition: all 0.2s ease; }
    .history-card:hover { border-color: #00CCFF; transform: translateX(5px); background: rgba(0, 204, 255, 0.05); }
    .hist-title { font-weight: 600; color: #FFFFFF; font-size: 1.1rem; margin: 0; }
    .hist-meta { font-family: 'Courier New', monospace; font-size: 0.85rem; color: #AAAAAA; margin-top: 4px; }

    .stButton > button { width: 100%; background-color: #00CCFF; color: #020c1a; font-weight: 700; border: none; padding: 0.75rem; transition: all 0.3s ease; }
    .stButton > button:hover { background-color: #33DFFF; transform: translateY(-2px); }
    .stButton > button:disabled { background-color: rgba(255, 255, 255, 0.2); color: #888888; }

    @keyframes beam-flow { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
    div[data-baseweb="slider"] div[role="progressbar"] { background: linear-gradient(90deg, #005f73, #00CCFF, #FFFFFF, #00CCFF, #005f73) !important; background-size: 200% 100% !important; animation: beam-flow 2.5s linear infinite !important; box-shadow: 0 0 10px rgba(0, 204, 255, 0.6); height: 8px !important; border-radius: 10px; }
    div[data-baseweb="slider"] > div > div:first-child { background: rgba(255, 255, 255, 0.25) !important; height: 8px !important; border-radius: 10px; }
    div[data-baseweb="slider"] div[role="slider"] { background-color: #020c1a !important; border: 2px solid #00CCFF !important; box-shadow: 0 0 10px rgba(0, 204, 255, 0.8) !important; width: 22px !important; height: 22px !important; }
    div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] { color: #00CCFF !important; font-family: monospace; }
    div[data-baseweb="slider"] p { color: #FFFFFF !important; }
    div[data-baseweb="select"] > div { background-color: rgba(0, 12, 26, 0.8) !important; border-color: rgba(0, 204, 255, 0.3) !important; }
    div[data-baseweb="tag"] { background-color: rgba(0, 204, 255, 0.2) !important; border: 1px solid #00CCFF !important; }
    div[data-baseweb="tag"] span { color: white !important; }
    [data-testid="stCheckbox"] label span, [data-testid="stRadio"] label span { color: #FFFFFF !important; font-weight: 500; }
    .timer-container { width: 100%; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; margin-top: 10px; overflow: hidden; }
    .timer-bar { height: 100%; background: linear-gradient(90deg, #00CCFF, #33DFFF); transition: width 0.1s linear; }
    [data-testid="stDataFrame"] { background: rgba(0, 0, 0, 0.2); border: 1px solid rgba(0, 204, 255, 0.3); border-radius: 10px; }
    [data-testid="stDataFrame"] th { background-color: rgba(0, 204, 255, 0.15) !important; border-bottom: 2px solid #00CCFF !important; color: white !important; }

    /* Specific styling for st.image in container to mimic video frame */
    [data-testid="stImage"] { border: 2px solid rgba(0, 204, 255, 0.3); border-radius: 8px; box-shadow: 0 0 20px rgba(0, 204, 255, 0.1); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    models = {}
    try:
        models['yolo'] = YOLO("yolov8n.pt")

        face_path = os.path.join(ROOT_DIR, "models", "face", "yolov8n-face-lindevs.pt")

        if os.path.exists(face_path):
            models['face'] = YOLO(face_path)
        else:
            models['face'] = None
            print(f"Face model not found at: {face_path}")

    except Exception as e:
        models['yolo'] = YOLO("yolov8n.pt")
        models['face'] = None
        print(f"Error loading models: {e}")
    return models


def detect_faces_yolo(image, face_model):
    if face_model is None:
        return []
    results = face_model.predict(image, conf=0.5, verbose=False)
    face_boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            face_boxes.append([x1, y1, x2, y2])
    return face_boxes


def apply_blur_cv2_smart(frame, box, intensity):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1: return frame

    roi = frame[y1:y2, x1:x2]
    k_size = (intensity * 2) + 1
    try:
        blurred_roi = cv2.GaussianBlur(roi, (k_size, k_size), 0)
        frame[y1:y2, x1:x2] = blurred_roi
    except:
        pass
    return frame


def save_snapshot(frame, run_id):
    snap_dir = os.path.join(ROOT_DIR, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"snap_{run_id}_{timestamp}.jpg"
    path = os.path.join(snap_dir, filename)
    cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return path


def display_detection_charts(df_logs):
    if not df_logs.empty:
        st.markdown(f"""
            <div class="section-header" style="border-color: #00CCFF; margin-top: 1.5rem;"> 
                {ICON_SEARCH} <span>Analytics Visualization</span>
            </div>
        """, unsafe_allow_html=True)

        chart_theme = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')

        col1, col2 = st.columns(2)

        with col1:
            counts = df_logs['Class'].value_counts().reset_index()
            counts.columns = ['Object', 'Count']
            fig1 = px.bar(counts, x='Object', y='Count',
                          title="Object Frequency",
                          color='Count',
                          color_continuous_scale='Blues')
            fig1.update_layout(**chart_theme)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.pie(df_logs, names='Class', title='Detection Ratio', hole=0.4,
                          color_discrete_sequence=px.colors.sequential.Blues_r)
            fig2.update_layout(**chart_theme)
            st.plotly_chart(fig2, use_container_width=True)


def generate_session_report(session_data):
    try:
        buffer = io.BytesIO()
        PAGE_W, PAGE_H = A4
        MARGIN_X = 0.6 * inch

        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * inch, bottomMargin=1.1 * inch, leftMargin=MARGIN_X,
                                rightMargin=MARGIN_X)
        COLOR_BG = colors.HexColor('#020c1a')
        COLOR_PANEL = colors.HexColor('#0b1d36')
        COLOR_NEON = colors.HexColor('#00CCFF')
        COLOR_TEAL = colors.HexColor('#0A9396')
        COLOR_TEXT = colors.white
        COLOR_DIM = colors.HexColor('#8899A6')

        def header_footer_gen(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(COLOR_BG)
            canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

            main_title = "DetectaX "
            sub_title = "Live Analysis"

            canvas.setFont("Helvetica-Bold", 24)
            canvas.setFillColor(COLOR_TEXT)
            canvas.drawString(MARGIN_X, PAGE_H - 55, main_title)
            canvas.setFont("Helvetica-Bold", 18)
            canvas.setFillColor(COLOR_NEON)
            canvas.drawString(MARGIN_X + canvas.stringWidth(main_title, "Helvetica-Bold", 24), PAGE_H - 55, sub_title)
            canvas.setStrokeColor(COLOR_NEON)
            canvas.setLineWidth(0.8)
            canvas.line(MARGIN_X, PAGE_H - 70, PAGE_W - MARGIN_X, PAGE_H - 70)
            canvas.setStrokeColor(COLOR_TEAL)
            canvas.line(MARGIN_X, 55, PAGE_W - MARGIN_X, 55)
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(COLOR_DIM)
            canvas.drawString(MARGIN_X, 40, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            canvas.drawRightString(PAGE_W - MARGIN_X, 40, f"Page {doc.page}")
            canvas.restoreState()

        styles = getSampleStyleSheet()
        style_h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=15,
                                  textColor=COLOR_NEON, spaceBefore=20, spaceAfter=12)
        story = [Spacer(1, 0.25 * inch), Paragraph("01 // SESSION METRICS", style_h1)]

        df = session_data['df']
        col_w = (PAGE_W - 2 * MARGIN_X) / 3
        kpi_data = [["DURATION", "FRAMES SCANNED", "OBJECTS FOUND"],
                    [f"{session_data['duration']:.1f}s", str(session_data['frames_count']), str(len(df))]]
        t_metrics = Table(kpi_data, colWidths=[col_w] * 3)
        t_metrics.setStyle(TableStyle(
            [('BACKGROUND', (0, 0), (-1, -1), COLOR_PANEL), ('TEXTCOLOR', (0, 0), (-1, 0), COLOR_DIM),
             ('TEXTCOLOR', (0, 1), (-1, 1), COLOR_NEON), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
             ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'), ('FONTSIZE', (0, 1), (-1, 1), 18),
             ('BOTTOMPADDING', (0, 1), (-1, 1), 12), ('BOX', (0, 0), (-1, -1), 0.4, COLOR_TEAL)]))
        story.append(t_metrics)
        story.append(Spacer(1, 22))

        if session_data['snapshots']:
            story.append(Paragraph("02 // CAPTURED SNAPSHOTS", style_h1))
            img_rows, current_row = [], []
            for snap in session_data['snapshots'][:4]:
                current_row.append(RLImage(snap['path'], width=3.2 * inch, height=2.4 * inch))
                if len(current_row) == 2: img_rows.append(current_row); current_row = []
            if current_row: img_rows.append(current_row)
            story.append(Table(img_rows))
            story.append(Spacer(1, 25))

        if not df.empty:
            story.append(PageBreak())
            story.append(Paragraph("03 // ANALYTICS VISUALIZATION", style_h1))
            chart_theme = dict(plot_bgcolor='#020c1a', paper_bgcolor='#020c1a', font=dict(color='white'))
            counts = df['Class'].value_counts().reset_index()
            counts.columns = ['Object', 'Count']
            fig1 = px.bar(counts, x="Object", y="Count", color_discrete_sequence=['#00CCFF'])
            fig1.update_layout(**chart_theme)
            fig2 = px.pie(df, names='Class', hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
            fig2.update_layout(**chart_theme)
            story.append(
                RLImage(io.BytesIO(fig1.to_image(format="png", width=820, height=350)), width=PAGE_W - 2 * MARGIN_X,
                        height=3.3 * inch))
            story.append(Spacer(1, 12))
            story.append(
                RLImage(io.BytesIO(fig2.to_image(format="png", width=820, height=350)), width=PAGE_W - 2 * MARGIN_X,
                        height=3.3 * inch))

        doc.build(story, onFirstPage=header_footer_gen, onLaterPages=header_footer_gen)
        return buffer.getvalue(), f"DetectaX_Live_{datetime.now().strftime('%H%M')}.pdf"
    except:
        return None, None


@st.dialog("Detailed Session Log", width="large")
def view_history_popup(session):
    m1, m2, m3 = st.columns(3)
    m1.metric("Duration", f"{session['duration']:.1f}s")
    m2.metric("Total Objects", len(session['df']))
    m3.metric("Snapshots", len(session['snapshots']))
    st.markdown("---")
    display_detection_charts(session['df'])
    st.markdown("---")
    if session['snapshots']:
        st.markdown("#### Snapshots")
        cols = st.columns(3)
        for i, snap in enumerate(session['snapshots']):
            with cols[i % 3]: st.image(snap['path'], caption=snap['time'])
    if not session['df'].empty:
        st.markdown("#### Detection Log")
        st.dataframe(session['df'], use_container_width=True)


def main():
    st.markdown('<div class="body-bg"></div>', unsafe_allow_html=True)
    render_navbar()

    if 'history' not in st.session_state: st.session_state.history = []
    if 'run_rt' not in st.session_state: st.session_state.run_rt = False
    if 'rt_logs' not in st.session_state: st.session_state.rt_logs = []
    if 'rt_snapshots' not in st.session_state: st.session_state.rt_snapshots = []
    if 'last_snap_time' not in st.session_state: st.session_state.last_snap_time = 0
    if 'show_stop_dialog' not in st.session_state: st.session_state.show_stop_dialog = False
    if 'start_time_ref' not in st.session_state: st.session_state.start_time_ref = 0
    if 'accumulated_time' not in st.session_state: st.session_state.accumulated_time = 0
    if 'temp_session_data' not in st.session_state: st.session_state.temp_session_data = None

    st.markdown(f"""
            <div class="main-header-container">
                <div class="icon-box"><i class="bi bi-camera-video" style="font-size: 3.5rem;"></i></div>
                <div class="text-box">
                    <h1>Real-Time Detection <i  style="font-size: 2rem; color: #00CCFF; animation: pulse-tech 1.5s infinite;"></i></h1>
                    <p>Advanced live stream detection with privacy filters and automated logging.</p>
                </div>
            </div>
            <hr style="border:0; height:1px; background:linear-gradient(to right, rgba(0,204,255,0), rgba(0,204,255,0.5), rgba(0,204,255,0)); margin-bottom:1.5rem;">
            <style>
                @keyframes pulse-tech {{
                    0% {{ opacity: 0.5; transform: scale(1); }}
                    50% {{ opacity: 1; transform: scale(1.1); text-shadow: 0 0 10px #00CCFF; }}
                    100% {{ opacity: 0.5; transform: scale(1); }}
                }}
            </style>
        """, unsafe_allow_html=True)

    @st.dialog("Session Interrupted")
    def stop_confirmation_dialog():
        st.write("The session has been paused. What would you like to do?")
        col_res, col_save, col_disc = st.columns(3)
        with col_res:
            if st.button("Resume", use_container_width=True):
                st.session_state.run_rt = True
                st.session_state.start_time_ref = time.time()
                st.session_state.show_stop_dialog = False
                st.rerun()
        with col_save:
            if st.button("End & Save", use_container_width=True):
                if st.session_state.temp_session_data:
                    st.session_state.history.append(st.session_state.temp_session_data)
                    st.success("Session Saved!")
                st.session_state.temp_session_data = None
                st.session_state.accumulated_time = 0
                st.session_state.run_rt = False
                st.session_state.show_stop_dialog = False
                st.rerun()
        with col_disc:
            if st.button("Discard", use_container_width=True):
                st.session_state.temp_session_data = None
                st.session_state.accumulated_time = 0
                st.session_state.run_rt = False
                st.session_state.show_stop_dialog = False
                st.rerun()

    if st.session_state.show_stop_dialog: stop_confirmation_dialog()

    col_video, col_data = st.columns([1.8, 1])

    with col_video:
        st.markdown('<div class="section-header"><i class="bi bi-broadcast"></i> Live Feed</div>',
                    unsafe_allow_html=True)
        video_container = st.empty()
        timer_placeholder = st.empty()

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            start_btn = st.button("START SESSION", disabled=st.session_state.run_rt, use_container_width=True)
        with c2:
            if st.button("STOP SESSION", disabled=not st.session_state.run_rt, use_container_width=True):
                st.session_state.run_rt = False
                st.session_state.accumulated_time += (time.time() - st.session_state.start_time_ref)
                st.session_state.show_stop_dialog = True
                st.rerun()

        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

        st.markdown(f'<div class="section-header">{ICON_SETTINGS} <span>Configuration</span></div>',
                    unsafe_allow_html=True)

        conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
        max_detections = st.slider("Maximum Detections", 1, 50, 20, 1)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">{ICON_FILTER} <span>Target Filter</span></div>',
                    unsafe_allow_html=True)

        COMMON_OBJECTS = ["Person", "Car", "Bus", "Truck", "Cell Phone", "Laptop", "Bottle", "Chair", "Traffic Light"]
        selected_filters = st.multiselect("Select objects to focus on (Empty = All)", options=COMMON_OBJECTS,
                                          default=[])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">{ICON_PRIVACY} <span>Privacy Mode</span></div>',
                    unsafe_allow_html=True)

        privacy_mode = st.selectbox("Select Privacy/Blurring Mode:",
                                    options=["None", "Blur Faces Only", "Blur Whole Person"], index=0)

        blur_intensity = 0
        if privacy_mode != "None":
            blur_intensity = st.slider("Blur Intensity", 5, 50, 15)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f'<div class="section-header">{ICON_REC} <span>Recording Settings</span></div>',
                    unsafe_allow_html=True)

        snapshot_interval = st.slider("Snapshot Interval (s)", 1, 10, 3,
                                      help="Time in seconds between automatic snapshots")

        col_tog1, col_tog2 = st.columns(2)
        with col_tog1:
            enable_snapshot = st.toggle("Auto-Capture", value=True)
        with col_tog2:
            show_boxes = st.toggle("Show Bounding Boxes", value=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_data:
        st.markdown('<div class="section-header"><i class="bi bi-graph-up-arrow"></i> Live Metrics</div>',
                    unsafe_allow_html=True)
        kpi_container = st.container(height=200, border=True)
        with kpi_container: kpi_placeholder = st.empty()

        st.markdown('<br><div class="section-header"><i class="bi bi-list-check"></i> Events Log</div>',
                    unsafe_allow_html=True)
        log_container = st.container(height=240, border=True)
        with log_container: log_placeholder = st.empty()

    if start_btn:
        st.session_state.run_rt = True
        st.session_state.show_stop_dialog = False
        st.session_state.rt_logs = []
        st.session_state.rt_snapshots = []
        st.session_state.temp_session_data = None
        st.session_state.accumulated_time = 0
        st.session_state.start_time_ref = time.time()
        st.rerun()

    if st.session_state.run_rt:
        models = load_models()
        model = models['yolo']
        cap = cv2.VideoCapture(0)


        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            st.error("Optical Sensor Unavailable!")
            st.session_state.run_rt = False
        else:
            frame_count = 0
            last_ui_update = time.time()


            while st.session_state.run_rt:
                ret, frame = cap.read()
                if not ret: break

                current_time = time.time()
                elapsed_in_this_run = current_time - st.session_state.start_time_ref
                total_elapsed = st.session_state.accumulated_time + elapsed_in_this_run


                if total_elapsed >= 60:
                    st.toast("Session Complete: 60s Limit Reached", icon="🏁")
                    session_summary = {"id": len(st.session_state.history) + 1,
                                       "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                       "duration": total_elapsed, "frames_count": frame_count,
                                       "snapshots": list(st.session_state.rt_snapshots),
                                       "df": pd.DataFrame(list(st.session_state.rt_logs))}
                    st.session_state.history.append(session_summary)
                    st.session_state.run_rt = False
                    st.session_state.accumulated_time = 0
                    st.rerun()
                    break


                remaining = 60 - total_elapsed
                progress = min(total_elapsed / 60.0, 1.0)
                timer_placeholder.markdown(
                    f"""<div style="display:flex; justify-content:space-between; color:#00CCFF; font-size:0.8rem; margin-bottom:2px; font-weight:bold;"><span><i class="bi bi-record-circle-fill" style="color:#FF4136;"></i> RECORDING</span><span>{remaining:.1f}s REMAINING</span></div><div class="timer-container"><div class="timer-bar" style="width: {progress * 100}%;"></div></div>""",
                    unsafe_allow_html=True)


                frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


                results = model.predict(rgb_frame, conf=conf_threshold, verbose=False)

                current_detections = []
                final_boxes_for_drawing = []
                filter_set = set(k.lower() for k in selected_filters)

                processed_frame = frame.copy()

                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    if len(boxes) > max_detections: boxes = boxes[:max_detections]
                    for box in boxes:
                        label = result.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        r = box.xyxy[0].astype(int)

                        if privacy_mode == "Blur Whole Person" and label.lower() == "person":
                            processed_frame = apply_blur_cv2_smart(processed_frame, r, blur_intensity)

                        if not filter_set or label.lower() in filter_set:

                            current_detections.append({"Timestamp": datetime.now().strftime("%H:%M:%S"), "Class": label,
                                                       "Confidence": conf, "BBox": r})
                            final_boxes_for_drawing.append((r, label, conf))

                if privacy_mode == "Blur Faces Only" and models['face']:
                    face_boxes = detect_faces_yolo(rgb_frame, models['face'])
                    for fb in face_boxes:
                        processed_frame = apply_blur_cv2_smart(processed_frame, fb, blur_intensity)

                if show_boxes:
                    for (r, label, conf) in final_boxes_for_drawing:
                        x1, y1, x2, y2 = r
                        color = (255, 204, 0)
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 1)
                        text = f"{label.upper()} {conf:.0%}"
                        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(processed_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                        cv2.putText(processed_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                final_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)


                video_container.image(final_rgb, channels="RGB", use_container_width=True)

                if current_detections:
                    log_entries = [{k: v for k, v in d.items() if k != 'BBox'} for d in current_detections]
                    st.session_state.rt_logs.extend(log_entries)
                    if enable_snapshot and (time.time() - st.session_state.last_snap_time > snapshot_interval):
                        path = save_snapshot(final_rgb, "auto")
                        st.session_state.rt_snapshots.append(
                            {"path": path, "time": datetime.now().strftime("%H:%M:%S")})
                        st.session_state.last_snap_time = time.time()

                frame_count += 1

                # Update Temp Session Data structure without creating full DF
                st.session_state.temp_session_data = {
                    "id": len(st.session_state.history) + 1,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": total_elapsed,
                    "frames_count": frame_count,
                    "snapshots": list(st.session_state.rt_snapshots),
                    "df": None  # Defer DataFrame creation until save/stop
                }

                # --- THROTTLED UI UPDATES (Only update every 0.5 seconds) ---
                if current_time - last_ui_update > 0.5:
                    with kpi_placeholder.container():
                        fps = frame_count / total_elapsed if total_elapsed > 0 else 0
                        k1, k2 = st.columns(2)
                        with k1:
                            st.markdown(
                                f"""<div class="summary-metric-card"><div class="label">FPS</div><div class="value">{fps:.1f}</div><div class="sub-text positive">Live</div></div>""",
                                unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                            blur_c = "positive" if privacy_mode != "None" else "neutral"
                            st.markdown(
                                f"""<div class="summary-metric-card"><div class="label">Privacy</div><div class="value" style="font-size:1rem;">{privacy_mode.split(' ')[0]}</div><div class="sub-text {blur_c}">Active</div></div>""",
                                unsafe_allow_html=True)
                        with k2:
                            st.markdown(
                                f"""<div class="summary-metric-card"><div class="label">Objects</div><div class="value">{len(current_detections)}</div><div class="sub-text neutral">Current</div></div>""",
                                unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown(
                                f"""<div class="summary-metric-card"><div class="label">Total Detected</div><div class="value" style="font-size:1.5rem;">{len(st.session_state.rt_logs)}</div><div class="sub-text neutral">Session</div></div>""",
                                unsafe_allow_html=True)

                    with log_placeholder.container():
                        if st.session_state.rt_logs:
                            # Only create a small slice DF for display
                            df_display = pd.DataFrame(st.session_state.rt_logs[-10:])
                            if not df_display.empty:
                                df_display = df_display.iloc[::-1][["Timestamp", "Class", "Confidence"]].copy()
                                df_display["Confidence"] = df_display["Confidence"].apply(lambda x: f"{x:.0%}")
                                st.dataframe(df_display, use_container_width=True, hide_index=True)
                        else:
                            st.caption("Waiting for detections...")

                    last_ui_update = current_time

            cap.release()

            # Finalize Session Data if stopped naturally
            if st.session_state.temp_session_data:
                st.session_state.temp_session_data["df"] = pd.DataFrame(list(st.session_state.rt_logs))

    else:
        video_container.markdown(
            f'''<div class="video-frame" style="height:400px; display:flex; align-items:center; justify-content:center; flex-direction:column; border-style:dashed; opacity:0.7;"><i class="bi bi-camera-video-off" style="font-size:3rem; color:#555;"></i><h4 style="color:#AAA; margin:10px 0;">Optical Sensor Offline</h4><small style="color:#555;">Click START SESSION to begin analysis</small></div>''',
            unsafe_allow_html=True)
        with kpi_container:
            st.info("Metrics will appear here during session")
        with log_container:
            st.info("Logs will appear here during session")

    st.markdown("---")
    st.markdown(f'<div class="section-header">{ICON_HISTORY} <span>Session History</span></div>',
                unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("No recorded sessions yet.")
    else:
        for session in reversed(st.session_state.history):
            with st.container():
                c_img, c_info, c_action = st.columns([1, 3, 1])
                with c_img:
                    if session['snapshots']:
                        st.image(session['snapshots'][0]['path'], use_container_width=True)
                    else:
                        st.markdown(
                            '<div style="height:80px; background:rgba(0,204,255,0.05); border:1px dashed #00CCFF; border-radius:4px;"></div>',
                            unsafe_allow_html=True)
                with c_info:
                    # Ensure DF exists for old sessions or edge cases
                    det_count = len(session['df']) if session['df'] is not None else 0
                    st.markdown(
                        f"""<div class="history-card"><div class="history-header"><span>SESSION LOG #{session['id']:02d}</span><span style="color:#FFF;">{session['duration']:.1f}s</span></div><div class="hist-meta"><i class="bi bi-calendar"></i> {session['timestamp']} &nbsp;|&nbsp; <i class="bi bi-images"></i> {len(session['snapshots'])} Snaps &nbsp;|&nbsp;<i class="bi bi-box"></i> {det_count} Detections</div></div>""",
                        unsafe_allow_html=True)
                with c_action:
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_pdf, col_view = st.columns(2)
                    with col_pdf:
                        if st.button("PDF", key=f"btn_pdf_{session['id']}", use_container_width=True):
                            with st.spinner("Generating Report..."):
                                pdf_bytes, fname = generate_session_report(session)
                                if pdf_bytes: st.download_button("Download", pdf_bytes, fname, "application/pdf",
                                                                 key=f"dl_{session['id']}", use_container_width=True)
                    with col_view:
                        if st.button("View", key=f"btn_view_{session['id']}",
                                     use_container_width=True): view_history_popup(session)

    render_footer()


if __name__ == "__main__":
    main()