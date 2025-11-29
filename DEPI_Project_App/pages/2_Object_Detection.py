import base64
import io
import json
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
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
import cv2
import zipfile
from ultralytics import YOLO

try:
    face_model = YOLO("DEPI_Project_App/models/face/yolov8n-face-lindevs.pt")
except:
    face_model = None

try:
    from navbar.navbar import render_navbar
    from footer.footer import render_footer
except ImportError:
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
        return [{"label": "Person", "confidence": 0.95, "bbox": [50, 50, 250, 450]},
                {"label": "Car", "confidence": 0.90, "bbox": [300, 100, 500, 350]},
                {"label": "Person", "confidence": 0.88, "bbox": [550, 60, 700, 400]}]


    def draw_bounding_boxes(image, detections):
        draw = ImageDraw.Draw(image)
        for det in detections:
            color = "#00CCFF"
            draw.rectangle(det['bbox'], outline=color, width=3)
            text = f"{det['label']} {det['confidence']:.0%}"
            bbox = draw.textbbox((0, 0), text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle(
                [det['bbox'][0], det['bbox'][1] - text_height - 4, det['bbox'][0] + text_width + 4, det['bbox'][1]],
                fill=color
            )
            draw.text((det['bbox'][0] + 2, det['bbox'][1] - text_height - 2), text, fill="black")
        return image


    CLASS_COLORS = {'default': '#00CCFF'}

ICON_TARGET = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-target"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>"""
ICON_UPLOAD = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-upload-cloud"><polyline points="16 16 12 12 8 16"></polyline><line x1="12" y1="12" x2="12" y2="21"></line><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path><polyline points="16 16 12 12 8 16"></polyline></svg>"""
ICON_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-settings"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>"""
ICON_SEARCH = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-search"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>"""
ICON_IMAGE = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>"""
ICON_ARROW_DOWN = """<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-arrow-down-circle"><circle cx="12" cy="12" r="10"></circle><polyline points="8 12 12 16 16 12"></polyline><line x1="12" y1="8" x2="12" y2="16"></line></svg>"""
ICON_FILTER = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-filter"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon></svg>"""
ICON_PRIVACY = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-shield"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>"""
ICON_HISTORY = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-clock"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>"""
ICON_CAMERA = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-camera"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>"""
ICON_CROP = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-crop"><path d="M6.13 1L6 16a2 2 0 0 0 2 2h15"></path><path d="M1 6.13L16 6a2 2 0 0 1 2 2v15"></path></svg>"""
ICON_LAYERS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-layers"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>"""

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
        pass


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

    .main-header-container { margin-top: -80px !important; margin-bottom: 40px !important; }
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

    .img-container.completed-effect::after {
        content: 'ANALYSIS COMPLETED';
        position: absolute; bottom: 20px; right: 20px;
        color: #00CCFF; font-family: monospace; font-weight: bold;
        background: rgba(0,0,0,0.8); padding: 5px 10px; border-radius: 4px;
        z-index: 11; border: 1px solid #00CCFF;
        box-shadow: 0 0 10px rgba(0, 204, 255, 0.5);
    }

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

    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        background-color: rgba(0, 204, 255, 0.03) !important;
        border: 2px dashed rgba(0, 204, 255, 0.5) !important;
        border-radius: 10px !important;
        padding: 1rem;
        transition: border 0.3s;
    }
    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #00CCFF !important;
        background-color: rgba(0, 204, 255, 0.1) !important;
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

    div[data-baseweb="select"] > div {
        background-color: rgba(0, 12, 26, 0.8) !important;
        border-color: rgba(0, 204, 255, 0.3) !important;
    }
    div[data-baseweb="tag"] {
        background-color: rgba(0, 204, 255, 0.2) !important;
        border: 1px solid #00CCFF !important;
    }
    div[data-baseweb="tag"] span {
        color: white !important;
    }

    [data-testid="stCheckbox"] label span, [data-testid="stRadio"] label span {
        color: #FFFFFF !important;
        font-weight: 500;
    }

    .history-card {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    .history-card:hover {
        border-color: #00CCFF;
        transform: translateY(-2px);
    }
    .history-header {
        font-weight: bold;
        color: #00CCFF;
        margin-bottom: 5px;
        display: flex;
        justify-content: space-between;
    }
    .history-meta {
        font-size: 0.8rem;
        color: #AAAAAA;
    }

    [data-testid="stTabs"] button {
        color: #FFFFFF !important;
        font-weight: bold;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        border-top-color: #00CCFF !important;
        color: #00CCFF !important;
    }
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
            st.info("No objects found matching criteria.")
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

        chart_theme = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')

        fig1 = px.bar(detection_stats, x='Object', y='Count',
                      title="Object Distribution",
                      color='Count',
                      color_continuous_scale='Blues')
        fig1.update_layout(**chart_theme)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df_detections, x='confidence',
                            title="Confidence Distribution",
                            nbins=10,
                            color_discrete_sequence=['#00CCFF'])
        fig2.update_layout(**chart_theme)
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.pie(df_detections, names='label', title='Object Frequency Ratio', hole=0.4,
                      color_discrete_sequence=px.colors.sequential.Blues_r)
        fig3.update_layout(**chart_theme)
        st.plotly_chart(fig3, use_container_width=True)


def validate_and_process_image(file):
    MAX_SIZE_MB = 200
    MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

    if file.size > MAX_SIZE_BYTES:
        show_custom_toast(f"File size too large! Please upload an image smaller than {MAX_SIZE_MB}MB.", "error")
        st.session_state.uploaded_file_id = None
        st.session_state.original_image_bytes = None
        st.session_state.results_ready = False
        return False

    try:
        img = Image.open(file)
        width, height = img.size
        MIN_DIMENSION = 100
        MAX_DIMENSION = 6000

        if width < MIN_DIMENSION or height < MIN_DIMENSION:
            show_custom_toast(f"Image resolution too low! Min dimensions: {MIN_DIMENSION}x{MIN_DIMENSION}px.", "error")
            st.session_state.uploaded_file_id = None
            st.session_state.original_image_bytes = None
            st.session_state.results_ready = False
            return False

        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            show_custom_toast(f"Image resolution too high! Max dimensions: {MAX_DIMENSION}x{MAX_DIMENSION}px.", "error")
            st.session_state.uploaded_file_id = None
            st.session_state.original_image_bytes = None
            st.session_state.results_ready = False
            return False

        file.seek(0)

        st.session_state.loading = False
        st.session_state.results_ready = False
        st.session_state.filtered_results = []
        st.session_state.processed_image_bytes = None
        st.session_state.error = None
        st.session_state.uploaded_file_id = getattr(file, 'file_id', str(time.time()))
        st.session_state.df_analytics = pd.DataFrame()
        st.session_state.inference_time = 0.0
        st.session_state.original_image_bytes = file.getvalue()
        return True

    except Exception as e:
        show_custom_toast(f"Invalid image file: {str(e)}", "error")
        return False


def file_uploader_callback():
    if 'browse_uploader' in st.session_state and st.session_state.browse_uploader is not None:
        files = st.session_state.browse_uploader
        if len(files) == 1:
            file = files[0]
            if getattr(file, 'file_id', str(time.time())) != st.session_state.uploaded_file_id:
                validate_and_process_image(file)


def camera_input_callback():
    if 'camera_capture' in st.session_state and st.session_state.camera_capture is not None:
        file = st.session_state.camera_capture
        current_bytes = file.getvalue()
        if st.session_state.original_image_bytes != current_bytes:
            validate_and_process_image(file)
            st.session_state.camera_enabled = False


def detect_faces_yolo(image):
    if face_model is None:
        return []
    results = face_model.predict(image, conf=0.65, verbose=False)
    face_boxes = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            face_boxes.append([x1, y1, x2, y2])

    return face_boxes


def apply_smart_face_blur(image, person_detections, intensity):
    img = image.copy()

    person_boxes = []
    if person_detections:
        for d in person_detections:
            if d['label'].lower() == 'person':
                person_boxes.append(d['bbox'])

    if not person_boxes:
        return img, False

    face_boxes = detect_faces_yolo(image)
    if not face_boxes:
        return img, False

    detection_occurred = False

    for (x1, y1, x2, y2) in face_boxes:
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2

        is_human_face = False

        for p_box in person_boxes:
            px1, py1, px2, py2 = p_box
            if (px1 <= face_center_x <= px2) and (py1 <= face_center_y <= py2):
                is_human_face = True
                break

        if not is_human_face:
            continue

        detection_occurred = True

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.width, x2)
        y2 = min(img.height, y2)

        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0: continue

        face_region = img.crop((x1, y1, x2, y2))
        blurred = face_region.filter(ImageFilter.GaussianBlur(radius=intensity / 3))

        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((0, 0, w, h), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=15))

        img.paste(blurred, (x1, y1), mask)

    return img, detection_occurred


def apply_general_object_blur(image, detections, intensity):
    img = image.copy()
    if not detections:
        return img, False

    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.width, x2)
        y2 = min(img.height, y2)

        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0: continue

        obj_region = img.crop((x1, y1, x2, y2))
        blurred = obj_region.filter(ImageFilter.GaussianBlur(radius=intensity / 2))

        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((0, 0, w, h), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=5))

        img.paste(blurred, (x1, y1), mask)

    return img, True


def create_analytics_df(detections, img_height):
    df_data = []
    for det in detections:
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
        return pd.DataFrame(df_data)
    else:
        return pd.DataFrame()


def generate_pdf_report(history_item=None):
    from reportlab.graphics.shapes import Drawing, Line
    from reportlab.lib.utils import ImageReader

    try:
        if history_item:
            detections = history_item['detections']
            orig_bytes = history_item['original']
            proc_bytes = history_item['processed']
            inference_time = history_item['inference_time']

            try:
                temp_img = Image.open(io.BytesIO(orig_bytes))
                img_height = temp_img.height
                df_analytics = create_analytics_df(detections, img_height)
            except:
                df_analytics = pd.DataFrame()

        else:
            detections = st.session_state.filtered_results
            orig_bytes = st.session_state.original_image_bytes
            proc_bytes = st.session_state.processed_image_bytes
            inference_time = st.session_state.inference_time
            df_analytics = st.session_state.df_analytics

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"DEPI_Report_{timestamp}.pdf"
        buffer = io.BytesIO()

        PAGE_W, PAGE_H = A4
        MARGIN_X = 0.6 * inch
        MARGIN_Y = 0.7 * inch

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=1.5 * inch,
            bottomMargin=1.1 * inch,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X
        )

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
            sub_title = "Image Analyzer"

            canvas.setFont("Helvetica-Bold", 24)
            canvas.setFillColor(COLOR_TEXT)
            canvas.drawString(MARGIN_X, PAGE_H - 55, main_title)

            canvas.setFont("Helvetica-Bold", 18)
            canvas.setFillColor(COLOR_NEON)
            canvas.drawString(
                MARGIN_X + canvas.stringWidth(main_title, "Helvetica-Bold", 24),
                PAGE_H - 55,
                sub_title
            )

            canvas.setStrokeColor(COLOR_NEON)
            canvas.setLineWidth(0.8)
            canvas.line(MARGIN_X, PAGE_H - 70, PAGE_W - MARGIN_X, PAGE_H - 70)

            canvas.setFont("Helvetica-Bold", 16)
            canvas.setFillColor(COLOR_TEXT)
            canvas.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 55, "OBJECT DETECTION REPORT")

            canvas.setStrokeColor(COLOR_TEAL)
            canvas.line(MARGIN_X, 55, PAGE_W - MARGIN_X, 55)

            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(COLOR_DIM)
            canvas.drawString(MARGIN_X, 40, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

            canvas.drawRightString(PAGE_W - MARGIN_X, 40, f"Page {doc.page}")

            canvas.setFont("Helvetica-Oblique", 7)
            canvas.drawCentredString(PAGE_W / 2, 25,
                                     "© 2025 DetectaX — All Rights Reserved")

            canvas.restoreState()

        styles = getSampleStyleSheet()

        style_h1 = ParagraphStyle(
            'H1',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=15,
            textColor=COLOR_NEON,
            spaceBefore=20,
            spaceAfter=12
        )

        style_txt = ParagraphStyle(
            'txt',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=COLOR_TEXT,
            leading=14,
            spaceAfter=6
        )

        story = []
        story.append(Spacer(1, 0.25 * inch))

        total_objects = len(detections) if detections else 0
        avg_conf_val = (
            f"{df_analytics['Confidence'].mean():.1%}"
            if not df_analytics.empty else "0.0%"
        )

        story.append(Paragraph("01 // EXECUTIVE SUMMARY", style_h1))

        col_w = (PAGE_W - 2 * MARGIN_X) / 3
        kpi_table = Table([
            ["TOTAL OBJECTS", "AVG CONFIDENCE", "PROCESS TIME"],
            [str(total_objects), avg_conf_val, f"{inference_time:.3f}s"]
        ], colWidths=[col_w, col_w, col_w])

        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLOR_PANEL),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLOR_DIM),
            ('TEXTCOLOR', (0, 1), (-1, 1), COLOR_NEON),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, 1), 18),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 12),
            ('BOX', (0, 0), (-1, -1), 0.4, COLOR_TEAL)
        ]))

        story.append(kpi_table)
        story.append(Spacer(1, 22))

        story.append(Paragraph("02 // VISUAL EVIDENCE", style_h1))

        if orig_bytes and proc_bytes:

            def prep_img(pil_img):
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                orig_w, orig_h = pil_img.size
                MAX_W = 240
                MAX_H = 180
                scale = min(MAX_W / orig_w, MAX_H / orig_h)
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                resized = pil_img.resize((new_w, new_h))
                buf = io.BytesIO()
                resized.save(buf, format='PNG')
                buf.seek(0)
                return RLImage(buf, width=new_w, height=new_h)

            img1 = prep_img(Image.open(io.BytesIO(orig_bytes)))
            img2 = prep_img(Image.open(io.BytesIO(proc_bytes)))

            gap_width = 20
            col_width = (PAGE_W - 2 * MARGIN_X - gap_width) / 2

            t_img = Table(
                [
                    [img1, "", img2],
                    [
                        Paragraph("<b><font color='#00CCFF'>RAW INPUT</font></b>", styles["BodyText"]),
                        "",
                        Paragraph("<b><font color='#00CCFF'>AI OUTPUT</font></b>", styles["BodyText"])
                    ]
                ],
                colWidths=[col_width, gap_width, col_width]
            )

            t_img.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 1), (-1, 1), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),

                ('BOX', (0, 0), (0, 0), 1.0, COLOR_NEON),
                ('BOX', (2, 0), (2, 0), 1.0, COLOR_NEON),
            ]))

            story.append(t_img)
            story.append(Spacer(1, 25))

        if detections and not df_analytics.empty:

            story.append(Paragraph("03 // DETAILED LOGS", style_h1))

            data = [["CLASS", "CONFIDENCE", "POSITION", "SIZE"]]

            for _, r in df_analytics.iterrows():
                data.append([
                    r["Label"].upper(),
                    f"{r['Confidence']:.1%}",
                    str(r["Position"]),
                    str(r["Size"])
                ])

            table_w = PAGE_W - 2 * MARGIN_X
            widths = [table_w * 0.35, table_w * 0.20, table_w * 0.25, table_w * 0.20]

            det_table = Table(data, colWidths=widths, repeatRows=1)

            det_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0, 1, 1, 0.15)),
                ('TEXTCOLOR', (0, 0), (-1, 0), COLOR_NEON),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),

                ('TEXTCOLOR', (0, 1), (-1, -1), COLOR_TEXT),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('LINEBELOW', (0, 0), (-1, -1), 0.3, colors.Color(1, 1, 1, 0.1)),
            ]))

            story.append(det_table)
            story.append(Spacer(1, 25))

        if not df_analytics.empty:
            story.append(PageBreak())
            story.append(Paragraph("04 // ANALYTICS VISUALIZATION", style_h1))

            try:
                df = df_analytics

                chart_theme = dict(
                    plot_bgcolor='#020c1a',
                    paper_bgcolor='#020c1a',
                    font=dict(color='white')
                )

                counts = df['Label'].value_counts().reset_index()
                counts.columns = ['Object', 'Count']

                fig1 = px.bar(
                    counts, x="Object", y="Count",
                    color_discrete_sequence=['#00CCFF']
                )
                fig1.update_layout(**chart_theme)

                detectax_colors = ['#00CCFF', '#0A9396', '#005F73', '#94D2BD']

                fig2 = px.pie(
                    df, names='Label', hole=0.4,
                    color_discrete_sequence=detectax_colors
                )
                fig2.update_layout(**chart_theme)

                img1 = fig1.to_image(format="png", width=820, height=350)
                img2 = fig2.to_image(format="png", width=820, height=350)

                story.append(RLImage(io.BytesIO(img1),
                                     width=PAGE_W - 2 * MARGIN_X,
                                     height=3.3 * inch))

                story.append(Spacer(1, 12))

                story.append(RLImage(io.BytesIO(img2),
                                     width=PAGE_W - 2 * MARGIN_X,
                                     height=3.3 * inch))

            except Exception as err:
                story.append(Paragraph(str(err), style_txt))

        doc.build(story, onFirstPage=header_footer_gen, onLaterPages=header_footer_gen)
        return buffer.getvalue(), filename

    except Exception as e:
        st.error(f"PDF Error: {str(e)}")
        return None, None


@st.dialog("Detailed Analysis Log", width="large")
def view_history_popup(item):
    st.markdown("""
        <style>
            .img-label {
                text-align: center; 
                color: #AAAAAA; 
                font-size: 0.9rem; 
                margin-bottom: 5px; 
                font-family: 'Courier New', monospace;
                letter-spacing: 1px;
            }
        </style>
    """, unsafe_allow_html=True)

    col_orig, col_proc = st.columns(2)

    with col_orig:
        st.markdown('<div class="img-label">[ SOURCE INPUT ]</div>', unsafe_allow_html=True)
        st.image(item["original"], use_container_width=True)
        st.download_button("Download Source", item["original"], f"original_source.png", "image/png",
                           key=f"dl_orig_popup_{item['timestamp']}", use_container_width=True)

    with col_proc:
        st.markdown('<div class="img-label">[ PROCESSED OUTPUT ]</div>', unsafe_allow_html=True)
        st.image(item["processed"], use_container_width=True)
        st.download_button("Download Result", item["processed"], f"processed_result.png", "image/png",
                           key=f"dl_proc_popup_{item['timestamp']}", use_container_width=True)

    st.markdown("---")

    st.markdown("#### Analysis Metrics")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    def tech_metric(label, value, color="#fff"):
        return f"""
        <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(0,204,255,0.2); padding:10px; border-radius:5px; text-align:center;">
            <div style="font-size:0.8rem; color:#888; text-transform:uppercase;">{label}</div>
            <div style="font-size:1.4rem; font-weight:bold; color:{color}; font-family:'Poppins';">{value}</div>
        </div>
        """

    with mcol1:
        st.markdown(tech_metric("Processing Time", f"{item['inference_time']:.3f}s", "#00CCFF"), unsafe_allow_html=True)
    with mcol2:
        st.markdown(tech_metric("Objects Found", str(len(item['detections'])), "#FFFFFF"), unsafe_allow_html=True)
    with mcol3:
        blur_txt = item.get('blur_mode', 'N/A')
        if blur_txt == "None": blur_txt = "Disabled"
        st.markdown(tech_metric("Privacy Protocol", blur_txt, "#AAAAAA"), unsafe_allow_html=True)
    with mcol4:
        intensity = item.get('blur_intensity', 0) if item.get('blur_mode') != "None" else 0
        st.markdown(tech_metric("Filter Intensity", f"{intensity}%", "#AAAAAA"), unsafe_allow_html=True)

    display_detection_charts(item['detections'])

    st.markdown("---")
    st.markdown("#### Report Generation")

    col_pdf, col_crop = st.columns(2)

    with col_pdf:
        if st.button("Generate & Download PDF", key=f"btn_gen_pdf_popup_{item['timestamp']}", use_container_width=True):
            with st.spinner("Generating Report from History..."):
                pdf_data, filename = generate_pdf_report(history_item=item)
                if pdf_data:
                    st.download_button(label="Download PDF Report", data=pdf_data, file_name=filename,
                                       mime="application/pdf", use_container_width=True,
                                       key=f"dl_pdf_popup_final_{item['timestamp']}")
                else:
                    st.error("Failed to generate PDF.")

    with col_crop:
        if item['detections']:
            btn_key = f"toggle_crop_{item['timestamp']}"
            if st.button("View & Crop Objects", key=btn_key, use_container_width=True):
                if f"show_{btn_key}" not in st.session_state:
                    st.session_state[f"show_{btn_key}"] = True
                else:
                    st.session_state[f"show_{btn_key}"] = not st.session_state[f"show_{btn_key}"]

            if st.session_state.get(f"show_{btn_key}", False):
                try:
                    img_pil = Image.open(io.BytesIO(item["original"]))
                    blur_mode = item.get('blur_mode', 'None')
                    blur_intensity = item.get('blur_intensity', 30)
                    detections = item['detections']

                    if blur_mode == "Blur Faces Only":
                        img_pil, _ = apply_smart_face_blur(img_pil, detections, blur_intensity)
                    elif blur_mode == "Blur Detected Objects":
                        img_pil, _ = apply_general_object_blur(img_pil, detections, blur_intensity)

                    render_crops_gallery_inline(img_pil, detections)
                except Exception as e:
                    st.error(f"Error preparing crops: {e}")
        else:
            st.button("No Objects to Crop", disabled=True, use_container_width=True,
                      key=f"btn_crop_disabled_{item['timestamp']}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Detection Data Logs")

    if item['detections']:
        df_dets = pd.DataFrame(item['detections'])
        df_dets['bbox_str'] = df_dets['bbox'].apply(lambda x: f"[{x[0]}, {x[1]}, {x[2]}, {x[3]}]")
        display_df = df_dets[['label', 'confidence', 'bbox_str']].copy()
        display_df.columns = ['Object Class', 'Confidence Score', 'Coordinates [x1, y1, x2, y2]']
        display_df['Confidence Score'] = display_df['Confidence Score'].apply(lambda x: f"{x:.2%}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("System reported 0 detections for this analysis.")


def get_image_base64(image_pil):
    buff = io.BytesIO()
    image_pil.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode()


def render_crops_gallery_inline(image_pil, detections):
    st.markdown("""
        <style>
            /* Custom Scrollbar for the gallery content */
            .crop-gallery-wrapper ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            .crop-gallery-wrapper ::-webkit-scrollbar-track {
                background: #020c1a; 
                border-radius: 4px;
            }
            .crop-gallery-wrapper ::-webkit-scrollbar-thumb {
                background: #00CCFF; 
                border-radius: 4px;
                border: 1px solid #020c1a;
            }
            .crop-gallery-wrapper ::-webkit-scrollbar-thumb:hover {
                background: #33DFFF;
                box-shadow: 0 0 10px rgba(0, 204, 255, 0.7);
            }

            /* Style for crop frames */
            .crop-image-container {
                width: 100%;
                height: 150px; 
                display: flex;
                align-items: center;
                justify-content: center;
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 6px;
                overflow: hidden;
                margin-bottom: 8px;
                border: 1px solid rgba(0, 204, 255, 0.1);
            }

            .crop-image-container img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain; 
                transition: transform 0.3s ease;
            }

            .crop-image-container:hover img {
                transform: scale(1.05);
            }

            .crop-label {
                text-align: center;
                font-weight: 600;
                color: #FFFFFF;
                margin-bottom: 2px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .crop-conf {
                text-align: center;
                font-size: 0.8rem;
                color: #00CCFF;
                margin-bottom: 8px;
            }
        </style>
    """, unsafe_allow_html=True)

    if not detections:
        st.info("No objects found to crop.")
        return

    try:
        zip_buffer = io.BytesIO()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"all_crops_{timestamp}.zip"

        crops_data = []

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for idx, det in enumerate(detections):
                bbox = det['bbox']
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image_pil.width, x2)
                y2 = min(image_pil.height, y2)

                if x2 > x1 and y2 > y1:
                    crop = image_pil.crop((x1, y1, x2, y2))

                    img_byte_arr = io.BytesIO()
                    crop.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()

                    label_clean = det['label'].replace(" ", "_")
                    crop_name = f"{idx + 1}_{label_clean}.png"

                    zip_file.writestr(crop_name, img_bytes)

                    crops_data.append({
                        "name": crop_name,
                        "image_pil": crop,
                        "bytes": img_bytes,
                        "label": det['label'],
                        "conf": det['confidence']
                    })

        st.markdown("<div class='crop-gallery-wrapper'>", unsafe_allow_html=True)
        st.download_button(
            label="Download All Crops as ZIP",
            data=zip_buffer.getvalue(),
            file_name=zip_filename,
            mime="application/zip",
            use_container_width=True,
            key=f"dl_all_zip_inline_{int(time.time())}"
        )

        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 15px 0;'>", unsafe_allow_html=True)

        crop_cols = st.columns(3)
        for i, crop_item in enumerate(crops_data):
            with crop_cols[i % 3]:
                with st.container(border=True):
                    b64_img = get_image_base64(crop_item['image_pil'])

                    st.markdown(f"""
                        <div class="crop-image-container">
                            <img src="data:image/png;base64,{b64_img}">
                        </div>
                        <div class="crop-label">{crop_item['label'].title()}</div>
                        <div class="crop-conf">{crop_item['conf']:.0%}</div>
                    """, unsafe_allow_html=True)

                    st.download_button(
                        label="Download",
                        data=crop_item['bytes'],
                        file_name=crop_item['name'],
                        mime="image/png",
                        key=f"dl_crop_inline_{i}_{int(time.time())}",
                        use_container_width=True
                    )
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error generating gallery: {e}")


def render_history_section():
    st.markdown("""
    <style>
        .history-card-container {
            background-color: rgba(2, 12, 26, 0.6);
            border: 1px solid rgba(0, 204, 255, 0.2);
            border-left: 3px solid #00CCFF;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            transition: all 0.3s ease;
        }
        .history-card-container:hover {
            background-color: rgba(0, 204, 255, 0.05);
            border-color: #00CCFF;
            transform: translateX(5px);
        }
        .hist-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            color: #FFFFFF;
            font-size: 1.1rem;
            margin: 0;
        }
        .hist-meta {
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            color: #AAAAAA;
            margin-top: 4px;
        }
        .hist-badge {
            background-color: rgba(0, 204, 255, 0.15);
            color: #00CCFF;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: bold;
            border: 1px solid rgba(0, 204, 255, 0.3);
        }
        .no-data-box {
            text-align: center;
            padding: 40px;
            border: 1px dashed rgba(255,255,255,0.2);
            border-radius: 8px;
            color: #666;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div class="section-header">{ICON_HISTORY} <span>Analysis History</span></div>',
                unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class="no-data-box">
            <p>No analysis records found in this session.</p>
            <small>Process an image to save results here.</small>
        </div>
        """, unsafe_allow_html=True)
        return

    for i, item in enumerate(reversed(st.session_state.history)):
        actual_index = len(st.session_state.history) - 1 - i

        with st.container():
            col_layout = st.columns([1.2, 4, 1.5])

            with col_layout[0]:
                if item.get("processed"):
                    st.image(item["processed"], use_container_width=True)

            with col_layout[1]:
                timestamp = item['timestamp']
                obj_count = len(item['detections'])
                blur_status = "Active" if item.get('blur_mode') != "None" else "Off"

                st.markdown(f"""
                <div style="padding-left: 10px;">
                    <div class="hist-title">SCAN ID: #{actual_index + 1:03d}</div>
                    <div class="hist-meta">{timestamp}</div>
                    <div style="margin-top: 8px;">
                        <span class="hist-badge">{obj_count} Objects</span>
                        <span class="hist-badge" style="margin-left:5px;">Blur: {blur_status}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_layout[2]:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("View Log", key=f"hist_btn_{actual_index}", use_container_width=True):
                    view_history_popup(item)

            st.markdown("<hr style='margin: 5px 0; opacity: 0.1;'>", unsafe_allow_html=True)


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
    if 'selected_classes_filter' not in st.session_state: st.session_state.selected_classes_filter = []

    if 'blur_mode' not in st.session_state: st.session_state.blur_mode = "None"
    if 'blur_intensity' not in st.session_state: st.session_state.blur_intensity = 30

    if 'history' not in st.session_state: st.session_state.history = []
    if 'view_history_index' not in st.session_state: st.session_state.view_history_index = None

    if 'camera_enabled' not in st.session_state: st.session_state.camera_enabled = False

    if 'batch_files' not in st.session_state: st.session_state.batch_files = []
    if 'batch_results' not in st.session_state: st.session_state.batch_results = None

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

        st.markdown("<br>", unsafe_allow_html=True)

        tab_upload, tab_cam = st.tabs(["Upload Image", "Use Camera"])

        with tab_upload:
            st.markdown('<div class="browse-button-only">', unsafe_allow_html=True)
            uploaded_files = st.file_uploader("Browse files or drag & drop (Max 5 files)",
                                              type=['jpg', 'jpeg', 'png', 'bmp'],
                                              accept_multiple_files=True,
                                              label_visibility="visible", key="browse_uploader",
                                              on_change=file_uploader_callback)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_cam:
            st.markdown("<br>", unsafe_allow_html=True)

            if not st.session_state.camera_enabled:

                st.markdown(f"""
                <div style="
                    border: 2px dashed rgba(0, 204, 255, 0.3); 
                    border-radius: 10px; 
                    background: rgba(0, 0, 0, 0.2); 
                    padding: 30px; 
                    text-align: center; 
                    margin-bottom: 15px;
                    display: flex; 
                    flex-direction: column; 
                    align-items: center;
                ">
                    <div style="color: #00CCFF; margin-bottom: 10px; opacity: 0.8;">
                        <i class="bi bi-camera-video-off" style="font-size: 3rem;"></i>
                    </div>
                    <h5 style="color: #FFFFFF; margin: 0; font-weight: 600;">Optical Sensor Offline</h5>
                    <p style="color: #AAAAAA; font-size: 0.85rem; margin-top: 5px;">System is ready to initialize video feed</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button("INITIALIZE CAMERA STREAM", use_container_width=True, key="btn_init_cam"):
                    st.session_state.camera_enabled = True
                    st.rerun()

            else:
                cam_img = st.camera_input("Optical Sensor Active", key="camera_capture", label_visibility="collapsed",
                                          on_change=camera_input_callback)

                st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

                if st.button("TERMINATE FEED", use_container_width=True, key="btn_term_cam"):
                    st.session_state.camera_enabled = False
                    st.rerun()

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

        st.markdown(f'<div class="section-header">{ICON_FILTER} <span>Target Filter</span></div>',
                    unsafe_allow_html=True)

        COMMON_OBJECTS = [
            "Person", "Car", "Bus", "Truck", "Traffic Light", "Stop Sign",
            "Bicycle", "Motorcycle", "Dog", "Cat", "Chair", "Potted Plant",
            "Laptop", "Cell Phone", "Book"
        ]

        selected_filters = st.multiselect(
            "Select objects to focus on (Empty = All)",
            options=COMMON_OBJECTS,
            default=[],
            help="Only display specific objects from the list."
        )
        st.session_state.selected_classes_filter = selected_filters

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f'<div class="section-header">{ICON_PRIVACY} <span>Privacy Mode</span></div>',
                    unsafe_allow_html=True)

        blur_option = st.radio(
            "Select Privacy/Blurring Mode:",
            options=["None", "Blur Faces Only", "Blur Detected Objects"],
            index=0,
            help="Choose 'Blur Faces' for privacy or 'Blur Detected Objects' to hide specifically detected items."
        )
        st.session_state.blur_mode = blur_option

        if blur_option != "None":
            st.session_state.blur_intensity = st.slider("Blur Intensity", 1, 100, 30,
                                                        help="Control how strong the blur effect is.")

        st.markdown("<br>", unsafe_allow_html=True)

        disable_btn = True
        is_batch = False

        if uploaded_files:
            if len(uploaded_files) > 5:
                st.error("Maximum 5 files allowed in batch mode.")
            elif len(uploaded_files) > 1:
                is_batch = True
                disable_btn = False
            elif len(uploaded_files) == 1:
                is_batch = False
                disable_btn = False
        elif st.session_state.original_image_bytes:
            disable_btn = False

        predict_btn = st.button("Detect Objects", disabled=disable_btn)

        if predict_btn:
            if is_batch:
                st.session_state.batch_files = uploaded_files
                st.session_state.loading = True
                st.session_state.results_ready = False
                st.session_state.batch_results = []
                st.rerun()
            else:
                st.session_state.loading = True
                st.session_state.results_ready = False
                st.session_state.filtered_results = []
                st.session_state.processed_image_bytes = None
                st.session_state.error = None
                st.session_state.df_analytics = pd.DataFrame()
                st.rerun()

    with input_frame:
        scan_class = "scan-effect" if st.session_state.loading else ""

        if uploaded_files and len(uploaded_files) > 1 and len(uploaded_files) <= 5:
            st.markdown(f"""
                <div style="height: 445px; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 2px dashed rgba(0, 204, 255, 0.5); border-radius: 10px; background: rgba(0, 12, 26, 0.8);">
                    <div style="color: #00CCFF; margin-bottom: 15px;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-layers"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
                    </div>
                    <h3 style="color: #FFFFFF; margin: 0;">Batch Mode Active</h3>
                    <p style="color: #AAAAAA; font-size: 1.1rem; margin: 5px 0;">{len(uploaded_files)} Images Selected</p>
                    <small style="color: #00CCFF;">Ready to process queue</small>
                </div>
                """, unsafe_allow_html=True)
        elif st.session_state.original_image_bytes:
            img_b64 = base64.b64encode(st.session_state.original_image_bytes).decode()
            st.markdown(
                f"""<div class="img-container {scan_class}"><img src="data:image/png;base64,{img_b64}" /></div>""",
                unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="height: 445px; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 2px dashed rgba(0, 204, 255, 0.3); border-radius: 10px; background: rgba(0, 0, 0, 0.2);">
                <div style="color: #00CCFF; margin-bottom: 15px;">{ICON_IMAGE}</div>
                <p style="color: #AAAAAA; font-size: 1rem; margin: 0;">Waiting for image...</p>
            </div>
            """, unsafe_allow_html=True)

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

        if st.session_state.loading and st.session_state.batch_files:
            with result_frame:
                progress_bar = st.progress(0)
                status_text = st.empty()

                batch_history_temp = []
                files_to_process = st.session_state.batch_files

                for i, file in enumerate(files_to_process):
                    status_text.markdown(f"""
                        <div style="text-align:center; padding:20px;">
                            <p style="color:#00CCFF; font-weight:bold;">Processing Image {i + 1}/{len(files_to_process)}</p>
                            <small>{file.name}</small>
                        </div>
                    """, unsafe_allow_html=True)

                    try:
                        file.seek(0)
                        file_bytes = file.getvalue()
                        img_for_processing = Image.open(io.BytesIO(file_bytes))
                        img_height = img_for_processing.height

                        start_time = time.time()
                        detections = detect_objects(img_for_processing, threshold=confidence_threshold)
                        end_time = time.time()
                        inf_time = end_time - start_time

                        if st.session_state.selected_classes_filter:
                            filter_set = set(k.lower() for k in st.session_state.selected_classes_filter)
                            detections = [d for d in detections if d['label'].lower() in filter_set]

                        filtered = detections[:max_detections]

                        processed_bytes_batch = None
                        if filtered or st.session_state.blur_mode == "Blur Faces Only":
                            result_img = img_for_processing.copy()
                            if st.session_state.blur_mode == "Blur Faces Only":
                                result_img, _ = apply_smart_face_blur(result_img, filtered,
                                                                      st.session_state.blur_intensity)
                            elif st.session_state.blur_mode == "Blur Detected Objects":
                                result_img, _ = apply_general_object_blur(result_img, filtered,
                                                                          st.session_state.blur_intensity)

                            annotated = draw_bounding_boxes(result_img, filtered)
                            buf = io.BytesIO()
                            annotated.save(buf, format="PNG")
                            processed_bytes_batch = buf.getvalue()
                        else:
                            buf = io.BytesIO()
                            img_for_processing.save(buf, format="PNG")
                            processed_bytes_batch = buf.getvalue()

                        history_item = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "original": file_bytes,
                            "processed": processed_bytes_batch,
                            "detections": filtered,
                            "inference_time": inf_time,
                            "blur_mode": st.session_state.blur_mode,
                            "blur_intensity": st.session_state.blur_intensity
                        }
                        st.session_state.history.append(history_item)
                        batch_history_temp.append(history_item)

                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")

                    progress_bar.progress((i + 1) / len(files_to_process))

                st.session_state.batch_results = batch_history_temp
                st.session_state.batch_files = []
                st.session_state.loading = False
                st.rerun()

        with result_frame:
            if st.session_state.loading and not st.session_state.batch_files:
                st.markdown(
                    """<div class="custom-loader">
                        <div class="loader-spinner"></div>
                        <p>Analyzing Image via Azure...</p>
                        <small>This may take a moment.</small>
                    </div>""",
                    unsafe_allow_html=True)

            elif st.session_state.batch_results:
                st.markdown("""
                    <div style="height:100%; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
                        <div style="color:#00CCFF; font-size:3rem; margin-bottom:10px;"><i class="bi bi-check-circle-fill"></i></div>
                        <h3 style="color:white;">Batch Complete!</h3>
                        <p style="color:#aaa;">Processed images are saved in history.</p>
                    </div>
                """, unsafe_allow_html=True)

            elif st.session_state.results_ready:
                if st.session_state.error:
                    show_custom_toast(f"Error: {st.session_state.error}", "error")
                elif st.session_state.processed_image_bytes:
                    img_b64 = base64.b64encode(st.session_state.processed_image_bytes).decode()
                    if not st.session_state.filtered_results:
                        show_custom_toast("No objects found matching your criteria.", "warning")
                    else:
                        show_custom_toast("Analysis Completed Successfully!", "success")
                        if st.session_state.blur_mode == "Blur Faces Only":
                            time.sleep(0.5)

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
                                    <small style="opacity: 0.8;margin-bottom: 60px;">Upload an image or use camera.</small>
                                </div>
                                """,
                    unsafe_allow_html=True)

        if st.session_state.batch_results:
            zip_buffer = io.BytesIO()
            timestamp_batch = datetime.now().strftime("%Y%m%d_%H%M%S")
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for idx, item in enumerate(st.session_state.batch_results):
                    if item.get("processed"):
                        file_name = f"result_{idx + 1}_{item['timestamp'].replace(':', '-')}.png"
                        zip_file.writestr(file_name, item['processed'])

            st.download_button(
                label="Download Batch Results (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"batch_results_{timestamp_batch}.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_batch_zip"
            )

            if st.button("Clear Batch Results", use_container_width=True):
                st.session_state.batch_results = None
                st.rerun()

        if st.session_state.results_ready and not st.session_state.error and st.session_state.processed_image_bytes and not st.session_state.batch_results:
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

            st.markdown("---")
            st.markdown(f'<div class="section-header">{ICON_CROP} <span>Object Gallery</span></div>',
                        unsafe_allow_html=True)

            if st.session_state.filtered_results:
                btn_key_main = "toggle_crop_main"
                if st.button("View & Download Cropped Objects", key=btn_key_main, use_container_width=True):
                    if f"show_{btn_key_main}" not in st.session_state:
                        st.session_state[f"show_{btn_key_main}"] = True
                    else:
                        st.session_state[f"show_{btn_key_main}"] = not st.session_state[f"show_{btn_key_main}"]

                if st.session_state.get(f"show_{btn_key_main}", False):
                    try:
                        img_for_crops = Image.open(io.BytesIO(st.session_state.original_image_bytes))

                        if st.session_state.blur_mode == "Blur Faces Only":
                            img_for_crops, _ = apply_smart_face_blur(img_for_crops, st.session_state.filtered_results,
                                                                     st.session_state.blur_intensity)
                        elif st.session_state.blur_mode == "Blur Detected Objects":
                            img_for_crops, _ = apply_general_object_blur(img_for_crops,
                                                                         st.session_state.filtered_results,
                                                                         st.session_state.blur_intensity)

                        render_crops_gallery_inline(img_for_crops, st.session_state.filtered_results)
                    except Exception as e:
                        st.error(f"Error preparing gallery: {e}")
            else:
                st.info("No objects detected to crop.")

        if st.session_state.results_ready and not st.session_state.error and not st.session_state.batch_results:
            if st.session_state.filtered_results:
                display_detection_summary(st.session_state.filtered_results)
            else:
                display_detection_summary([])

    if st.session_state.loading and not st.session_state.batch_files:
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

            if st.session_state.selected_classes_filter:
                filter_set = set(k.lower() for k in st.session_state.selected_classes_filter)
                detections = [d for d in detections if d['label'].lower() in filter_set]

            filtered = detections[:max_detections]
            st.session_state.filtered_results = filtered

            st.session_state.df_analytics = create_analytics_df(filtered, img_height)

            if filtered or st.session_state.blur_mode == "Blur Faces Only":
                result_img = img_for_processing.copy()

                if st.session_state.blur_mode == "Blur Faces Only":
                    result_img, _ = apply_smart_face_blur(result_img, filtered, st.session_state.blur_intensity)

                elif st.session_state.blur_mode == "Blur Detected Objects":
                    result_img, _ = apply_general_object_blur(result_img, filtered, st.session_state.blur_intensity)

                annotated = draw_bounding_boxes(result_img, filtered)

                buf = io.BytesIO()
                annotated.save(buf, format="PNG")
                st.session_state.processed_image_bytes = buf.getvalue()
            else:
                buf = io.BytesIO()
                img_for_processing.save(buf, format="PNG")
                st.session_state.processed_image_bytes = buf.getvalue()

            history_item = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "original": st.session_state.original_image_bytes,
                "processed": st.session_state.processed_image_bytes,
                "detections": st.session_state.filtered_results,
                "inference_time": st.session_state.inference_time,
                "blur_mode": st.session_state.blur_mode,
                "blur_intensity": st.session_state.blur_intensity
            }
            st.session_state.history.append(history_item)

        except Exception as e:
            st.session_state.error = str(e)

        st.session_state.loading = False
        st.session_state.results_ready = True
        st.rerun()

    if st.session_state.results_ready and not st.session_state.error and not st.session_state.batch_results:
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

    render_history_section()
    render_footer()


if __name__ == "__main__":
    main()