import base64
import io
import json
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import plotly.express as px
import os
import time
from reportlab.lib.pagesizes import A4
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

ICON_TAGS = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" fill="currentColor" class="bi bi-tags-fill" viewBox="0 0 16 16"><path d="M2 2a1 1 0 0 1 1-1h4.586a1 1 0 0 1 .707.293l7 7a1 1 0 0 1 0 1.414l-4.586 4.586a1 1 0 0 1-1.414 0l-7-7A1 1 0 0 1 2 6.586V2zm3.5 4a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3z"/><path d="M1.293 7.793A1 1 0 0 1 1 7.086V2a1 1 0 0 0-1 1v4.586a1 1 0 0 0 .293.707l7 7a1 1 0 0 0 1.414 0l.043-.043-7.457-7.457z"/></svg>"""
ICON_UPLOAD = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-upload-cloud"><polyline points="16 16 12 12 8 16"></polyline><line x1="12" y1="12" x2="12" y2="21"></line><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path><polyline points="16 16 12 12 8 16"></polyline></svg>"""
ICON_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-settings"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>"""
ICON_RESULTS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-activity"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>"""
ICON_RESULTS_PLACEHOLDER = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" class="feather feather-activity"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>"""
ICON_ARROW_DOWN = """<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-arrow-down-circle"><circle cx="12" cy="12" r="10"></circle><polyline points="8 12 12 16 16 12"></polyline><line x1="12" y1="8" x2="12" y2="16"></line></svg>"""
ICON_STATS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-pie-chart"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path><path d="M22 12A10 10 0 0 0 12 2v10z"></path></svg>"""
ICON_LIST = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-list"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></svg>"""
ICON_IMAGE = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>"""
ICON_PRIVACY = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-shield"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>"""
ICON_HISTORY = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-clock"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>"""
ICON_LAYERS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-layers"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>"""

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

    body, html, button, a, div, span, input { 
        cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="%2300CCFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path><circle cx="7" cy="7" r="2"></circle></svg>') 2 2, auto !important; 
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
    .img-container.completed-effect { border: 2px solid #00CCFF; box-shadow: 0 0 15px rgba(0, 204, 255, 0.4); }

.img-container.completed-effect::after {
    content: 'ANALYSIS COMPLETED';
    position: absolute; bottom: 20px; right: 20px;
    color: #00CCFF; font-family: monospace; font-weight: bold;
    background: rgba(0,0,0,0.8); padding: 5px 10px; border-radius: 4px;
    z-index: 11; border: 1px solid #00CCFF;
    box-shadow: 0 0 10px rgba(0, 204, 255, 0.5); 
}

    @keyframes grid-move { 0% { background-position: 0 0; } 100% { background-position: 40px 40px; } }
    @keyframes pulse-border { from { box-shadow: 0 0 5px rgba(0, 204, 255, 0.3); } to { box-shadow: 0 0 20px rgba(0, 204, 255, 0.6); } }

    .prob-card {
        background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 12px 15px; margin-bottom: 10px; transition: all 0.3s ease; position: relative; overflow: hidden;
    }
    .prob-card:hover { border-color: #00CCFF; transform: translateX(5px); background: rgba(0, 204, 255, 0.08); box-shadow: -5px 0 15px rgba(0, 204, 255, 0.1); }
    .prob-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .prob-class { color: #FFFFFF; font-weight: 600; font-size: 1rem; display: flex; align-items: center; gap: 10px; }
    .prob-rank { color: #00CCFF; font-family: monospace; font-size: 0.9rem; opacity: 0.7; background: rgba(0, 204, 255, 0.1); padding: 2px 6px; border-radius: 4px; }
    .prob-score { color: #00CCFF; font-family: monospace; font-weight: 700; }
    .prob-bar-bg { height: 6px; width: 100%; background: rgba(255, 255, 255, 0.1); border-radius: 10px; overflow: hidden; }
    .prob-bar-fill { height: 100%; background: linear-gradient(90deg, #005f73, #00CCFF); border-radius: 10px; box-shadow: 0 0 10px rgba(0, 204, 255, 0.5); animation: loadBar 1.5s ease-out forwards; }
    @keyframes loadBar { from { width: 0; } }

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
    .top-prediction-card .confidence-value { color: #00CCFF; font-size: 2rem; margin: 0; font-weight: 600; }

    div[data-baseweb="slider"] div[role="progressbar"] { background: linear-gradient(90deg, #005f73, #00CCFF, #FFFFFF, #00CCFF, #005f73) !important; background-size: 200% 100% !important; animation: beam-flow 2.5s linear infinite !important; box-shadow: 0 0 10px rgba(0, 204, 255, 0.6); height: 8px !important; border-radius: 10px; }
    div[data-baseweb="slider"] > div > div:first-child { background: rgba(255, 255, 255, 0.25) !important; height: 8px !important; border-radius: 10px; }
    div[data-baseweb="slider"] div[role="slider"] { background-color: #020c1a !important; border: 2px solid #00CCFF !important; box-shadow: 0 0 10px rgba(0, 204, 255, 0.8) !important; width: 22px !important; height: 22px !important; }
    @keyframes beam-flow { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }

    [data-testid="stRadio"] label span { color: #FFFFFF !important; font-weight: 500; }
    [data-testid="stTabs"] button { color: #FFFFFF !important; font-weight: bold; }
    [data-testid="stTabs"] button[aria-selected="true"] { border-top-color: #00CCFF !important; color: #00CCFF !important; }
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


def detect_faces_yolo(image):
    if face_model is None:
        return []
    results = face_model.predict(image, conf=0.75, iou=0.5, verbose=False)
    face_boxes = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                face_boxes.append([x1, y1, x2, y2])
    return face_boxes


def apply_smart_face_blur(image, intensity):
    img = image.copy()
    face_boxes = detect_faces_yolo(image)
    if not face_boxes:
        return img, False
    for (x1, y1, x2, y2) in face_boxes:
        x1 = max(0, x1);
        y1 = max(0, y1);
        x2 = min(img.width, x2);
        y2 = min(img.height, y2)
        w = x2 - x1;
        h = y2 - y1
        if w <= 0 or h <= 0: continue
        face_region = img.crop((x1, y1, x2, y2))
        blurred = face_region.filter(ImageFilter.GaussianBlur(radius=intensity / 3))
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((0, 0, w, h), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=15))
        img.paste(blurred, (x1, y1), mask)
    return img, True


def generate_full_classification_report(history_item=None):
    try:
        if history_item:
            original_bytes = history_item['original']
            res = history_item['class_result']
            chart_df = history_item['chart_data']
            inf_time = history_item['inference_time']
            processed_bytes = history_item.get('processed', None)
        else:
            original_bytes = st.session_state.original_image_bytes
            res = st.session_state.classification_result
            chart_df = st.session_state.chart_data
            inf_time = st.session_state.inference_time
            processed_bytes = st.session_state.processed_image_bytes

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"DEPI_Clf_Report_{timestamp}.pdf"
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
            sub_title = "Image Classifier"

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
            canvas.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 55, "CLASSIFICATION REPORT")

            canvas.setStrokeColor(COLOR_TEAL)
            canvas.line(MARGIN_X, 55, PAGE_W - MARGIN_X, 55)

            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(COLOR_DIM)
            canvas.drawString(MARGIN_X, 40, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            canvas.drawRightString(PAGE_W - MARGIN_X, 40, f"Page {doc.page}")

            canvas.setFont("Helvetica-Oblique", 7)
            canvas.drawCentredString(PAGE_W / 2, 25, "© 2025 DetectaX — All Rights Reserved")

            canvas.restoreState()

        styles = getSampleStyleSheet()
        style_h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=15,
                                  textColor=COLOR_NEON, spaceBefore=20, spaceAfter=12)
        style_txt = ParagraphStyle('txt', parent=styles['Normal'], fontName='Helvetica', fontSize=10,
                                   textColor=COLOR_TEXT, leading=14, spaceAfter=6)

        story = []
        story.append(Spacer(1, 0.25 * inch))

        story.append(Paragraph("01 // EXECUTIVE SUMMARY", style_h1))
        col_w = (PAGE_W - 2 * MARGIN_X) / 3
        kpi_table = Table([
            ["PREDICTED CLASS", "CONFIDENCE SCORE", "PROCESS TIME"],
            [res['class'].title(), f"{res['confidence']:.2%}", f"{inf_time:.3f}s"]
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
        if original_bytes:
            def prep_img(pil_img):
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                orig_w, orig_h = pil_img.size
                MAX_W = 240;
                MAX_H = 180
                scale = min(MAX_W / orig_w, MAX_H / orig_h)
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                resized = pil_img.resize((new_w, new_h))
                buf = io.BytesIO()
                resized.save(buf, format='PNG')
                buf.seek(0)
                return RLImage(buf, width=new_w, height=new_h)

            img1 = prep_img(Image.open(io.BytesIO(original_bytes)))
            if processed_bytes:
                img2 = prep_img(Image.open(io.BytesIO(processed_bytes)))
                label2 = "PRIVACY MODE OUTPUT"
            else:
                img2 = img1
                label2 = "AI OUTPUT (SAME)"

            gap_width = 20
            col_width = (PAGE_W - 2 * MARGIN_X - gap_width) / 2
            t_img = Table([[img1, "", img2],
                           [Paragraph("<b><font color='#00CCFF'>RAW INPUT</font></b>", styles["BodyText"]), "",
                            Paragraph(f"<b><font color='#00CCFF'>{label2}</font></b>", styles["BodyText"])]],
                          colWidths=[col_width, gap_width, col_width])
            t_img.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                       ('BOTTOMPADDING', (0, 1), (-1, 1), 8), ('TOPPADDING', (0, 0), (-1, 0), 6),
                                       ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                                       ('BOX', (0, 0), (0, 0), 1.0, COLOR_NEON),
                                       ('BOX', (2, 0), (2, 0), 1.0, COLOR_NEON)]))
            story.append(t_img)
            story.append(Spacer(1, 25))

        if not chart_df.empty:
            story.append(Paragraph("03 // TOP PREDICTIONS", style_h1))
            data = [["RANK", "CLASS LABEL", "CONFIDENCE"]]
            for idx, row in chart_df.iterrows():
                data.append([f"#{idx + 1}", row['Class'].title(), f"{row['Confidence']:.2%}"])
            table_w = PAGE_W - 2 * MARGIN_X
            widths = [table_w * 0.15, table_w * 0.55, table_w * 0.30]
            det_table = Table(data, colWidths=widths, repeatRows=1)
            det_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.Color(0, 1, 1, 0.15)),
                                           ('TEXTCOLOR', (0, 0), (-1, 0), COLOR_NEON),
                                           ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                           ('TEXTCOLOR', (0, 1), (-1, -1), COLOR_TEXT),
                                           ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                           ('LINEBELOW', (0, 0), (-1, -1), 0.3, colors.Color(1, 1, 1, 0.1))]))
            story.append(det_table)
            story.append(Spacer(1, 25))

        if not chart_df.empty:
            story.append(PageBreak())
            story.append(Paragraph("04 // ANALYTICS VISUALIZATION", style_h1))
            try:
                df = chart_df
                chart_theme = dict(plot_bgcolor='#020c1a', paper_bgcolor='#020c1a', font=dict(color='white'))
                fig1 = px.bar(df, x='Class', y='Confidence', title=None, color='Class',
                              color_discrete_sequence=['#00CCFF', '#0A9396', '#005F73', '#94D2BD'])
                fig1.update_layout(**chart_theme, showlegend=False)
                fig2 = px.pie(df, names='Class', values='Confidence', title=None, hole=0.4,
                              color_discrete_sequence=['#00CCFF', '#0A9396', '#005F73', '#94D2BD'])
                fig2.update_layout(**chart_theme)
                img1 = fig1.to_image(format="png", width=820, height=350)
                img2 = fig2.to_image(format="png", width=820, height=350)
                story.append(RLImage(io.BytesIO(img1), width=PAGE_W - 2 * MARGIN_X, height=3.3 * inch))
                story.append(Spacer(1, 12))
                story.append(RLImage(io.BytesIO(img2), width=PAGE_W - 2 * MARGIN_X, height=3.3 * inch))
            except Exception as err:
                story.append(Paragraph(str(err), style_txt))

        doc.build(story, onFirstPage=header_footer_gen, onLaterPages=header_footer_gen)
        return buffer.getvalue(), filename
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None, None


def validate_and_process_image(file):
    MAX_SIZE_MB = 200
    MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024
    if file.size > MAX_SIZE_BYTES:
        show_custom_toast(f"File size too large! Please upload an image smaller than {MAX_SIZE_MB}MB.", "error")
        return False
    try:
        img = Image.open(file)
        width, height = img.size
        MIN_DIMENSION = 100;
        MAX_DIMENSION = 6000
        if width < MIN_DIMENSION or height < MIN_DIMENSION:
            show_custom_toast(f"Image resolution too low! Min dimensions: {MIN_DIMENSION}x{MIN_DIMENSION}px.", "error")
            return False
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            show_custom_toast(f"Image resolution too high! Max dimensions: {MAX_DIMENSION}x{MAX_DIMENSION}px.", "error")
            return False
        file.seek(0)
        st.session_state.loading = False
        st.session_state.results_ready = False
        st.session_state.classification_result = {}
        st.session_state.chart_data = pd.DataFrame()
        st.session_state.error = None
        st.session_state.inference_time = 0.0
        st.session_state.processed_image_bytes = None
        st.session_state.uploaded_file_id = getattr(file, 'file_id', str(time.time()))
        st.session_state.original_image_bytes = file.getvalue()
        return True
    except Exception as e:
        show_custom_toast(f"Invalid image file: {str(e)}", "error")
        return False


def file_uploader_callback():
    if 'browse_uploader_clf' in st.session_state and st.session_state.browse_uploader_clf is not None:
        files = st.session_state.browse_uploader_clf
        if len(files) == 1:
            file = files[0]
            if getattr(file, 'file_id', str(time.time())) != st.session_state.uploaded_file_id:
                validate_and_process_image(file)


def camera_input_callback():
    if 'camera_capture_clf' in st.session_state and st.session_state.camera_capture_clf is not None:
        file = st.session_state.camera_capture_clf
        current_bytes = file.getvalue()
        if st.session_state.original_image_bytes != current_bytes:
            validate_and_process_image(file)
            st.session_state.camera_enabled = False


@st.dialog("Detailed Analysis Log", width="large")
def view_history_popup(item):
    st.markdown(
        """<style>.img-label { text-align: center; color: #AAAAAA; font-size: 0.9rem; margin-bottom: 5px; font-family: 'Courier New', monospace; letter-spacing: 1px; }</style>""",
        unsafe_allow_html=True)
    col_orig, col_proc = st.columns(2)
    with col_orig:
        st.markdown('<div class="img-label">[ SOURCE INPUT ]</div>', unsafe_allow_html=True)
        st.image(item["original"], use_container_width=True)
    with col_proc:
        st.markdown('<div class="img-label">[ PROCESSED RESULT ]</div>', unsafe_allow_html=True)
        display_img = item.get("processed") if item.get("processed") else item["original"]
        st.image(display_img, use_container_width=True)
    st.markdown("---")
    st.markdown("#### Analysis Metrics")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    def tech_metric(label, value, color="#fff"):
        return f"""<div style="background:rgba(255,255,255,0.05); border:1px solid rgba(0,204,255,0.2); padding:10px; border-radius:5px; text-align:center;"><div style="font-size:0.8rem; color:#888; text-transform:uppercase;">{label}</div><div style="font-size:1.4rem; font-weight:bold; color:{color}; font-family:'Poppins';">{value}</div></div>"""

    res = item['class_result']
    with mcol1:
        st.markdown(tech_metric("Processing Time", f"{item['inference_time']:.3f}s", "#00CCFF"), unsafe_allow_html=True)
    with mcol2:
        st.markdown(tech_metric("Predicted Class", res['class'].title(), "#FFFFFF"), unsafe_allow_html=True)
    with mcol3:
        st.markdown(tech_metric("Confidence", f"{res['confidence']:.1%}", "#00CCFF"), unsafe_allow_html=True)
    with mcol4:
        blur_txt = item.get('blur_mode', 'None')
        if blur_txt == "None": blur_txt = "Disabled"
        st.markdown(tech_metric("Privacy Protocol", blur_txt, "#AAAAAA"), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### Visual Analytics")
    if not item['chart_data'].empty:
        df = item['chart_data']
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_bar = px.bar(df, x='Class', y='Confidence', title="Confidence Distribution", color='Confidence',
                             color_continuous_scale='Blues', text_auto='.1%')
            fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_c2:
            fig_pie = px.pie(df, names='Class', values='Confidence', title="Probability Ratio",
                             color_discrete_sequence=px.colors.sequential.Teal)
            fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("---")
    st.markdown("#### Report Generation")
    if st.button("Generate & Download PDF", key="btn_gen_pdf_popup_clf", use_container_width=True):
        with st.spinner("Generating Report from History..."):
            pdf_data, filename = generate_full_classification_report(history_item=item)
            if pdf_data:
                st.download_button(label="Download PDF Report", data=pdf_data, file_name=filename,
                                   mime="application/pdf", use_container_width=True, key="dl_pdf_popup_final_clf")


def render_history_section():
    st.markdown(
        """<style>.history-card-container { background-color: rgba(2, 12, 26, 0.6); border: 1px solid rgba(0, 204, 255, 0.2); border-left: 3px solid #00CCFF; padding: 15px; border-radius: 0 8px 8px 0; transition: all 0.3s ease; } .history-card-container:hover { background-color: rgba(0, 204, 255, 0.05); border-color: #00CCFF; transform: translateX(5px); } .hist-title { font-family: 'Poppins', sans-serif; font-weight: 600; color: #FFFFFF; font-size: 1.1rem; margin: 0; } .hist-meta { font-family: 'Courier New', monospace; font-size: 0.85rem; color: #AAAAAA; margin-top: 4px; } .hist-badge { background-color: rgba(0, 204, 255, 0.15); color: #00CCFF; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; border: 1px solid rgba(0, 204, 255, 0.3); } .no-data-box { text-align: center; padding: 40px; border: 1px dashed rgba(255,255,255,0.2); border-radius: 8px; color: #666; } </style>""",
        unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f'<div class="section-header">{ICON_HISTORY} <span>Analysis History</span></div>',
                unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown(
            """<div class="no-data-box"><p>No analysis records found in this session.</p><small>Process an image to save results here.</small></div>""",
            unsafe_allow_html=True)
        return
    for i, item in enumerate(reversed(st.session_state.history)):
        actual_index = len(st.session_state.history) - 1 - i
        with st.container():
            col_layout = st.columns([1.2, 4, 1.5])
            with col_layout[0]:
                display_img = item.get("processed") if item.get("processed") else item["original"]
                st.image(display_img, use_container_width=True)
            with col_layout[1]:
                timestamp = item['timestamp']
                res = item['class_result']
                blur_status = "Active" if item.get('blur_mode') != "None" else "Off"
                st.markdown(
                    f"""<div style="padding-left: 10px;"><div class="hist-title">SCAN ID: #{actual_index + 1:03d}</div><div class="hist-meta">{timestamp}</div><div style="margin-top: 8px;"><span class="hist-badge">{res['class'].title()} ({res['confidence']:.1%})</span><span class="hist-badge" style="margin-left:5px;">Blur: {blur_status}</span></div></div>""",
                    unsafe_allow_html=True)
            with col_layout[2]:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("View Log", key=f"hist_btn_clf_{actual_index}", use_container_width=True):
                    view_history_popup(item)
            st.markdown("<hr style='margin: 5px 0; opacity: 0.1;'>", unsafe_allow_html=True)


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
    if 'processed_image_bytes' not in st.session_state: st.session_state.processed_image_bytes = None
    if 'camera_enabled' not in st.session_state: st.session_state.camera_enabled = False
    if 'history' not in st.session_state: st.session_state.history = []
    if 'blur_mode' not in st.session_state: st.session_state.blur_mode = "None"
    if 'blur_intensity' not in st.session_state: st.session_state.blur_intensity = 30

    if 'batch_files_clf' not in st.session_state: st.session_state.batch_files_clf = []
    if 'batch_results_clf' not in st.session_state: st.session_state.batch_results_clf = None

    st.markdown(
        f"""<div class="main-header-container"><div class="icon-box">{ICON_TAGS}</div><div class="text-box"><h1>Image Classification</h1><p>Upload an image (or up to 5) to classify content into predefined categories.</p></div></div><hr class="styled-hr">""",
        unsafe_allow_html=True)

    col1, spacer, col2 = st.columns([1, 0.15, 1])
    FRAME_HEIGHT = 480

    col1, spacer, col2 = st.columns([1, 0.15, 1])
    FRAME_HEIGHT = 480

    with col1:
        st.markdown(f'<div class="section-header">{ICON_UPLOAD} <span>Input Image</span></div>', unsafe_allow_html=True)
        input_frame = st.container(border=True, height=FRAME_HEIGHT)
        st.markdown("<br>", unsafe_allow_html=True)

        tab_upload, tab_cam = st.tabs(["Upload Image", "Use Camera"])
        with tab_upload:
            st.markdown('<div class="browse-button-only">', unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Browse files or drag & drop (Max 5 files)",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True,
                label_visibility="visible",
                key="browse_uploader_clf",
                on_change=file_uploader_callback
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_cam:
            st.markdown("<br>", unsafe_allow_html=True)
            if not st.session_state.camera_enabled:
                st.markdown(
                    f"""<div style="border: 2px dashed rgba(0, 204, 255, 0.3); 
                    border-radius: 10px; background: rgba(0, 0, 0, 0.2); 
                    padding: 30px; text-align: center; margin-bottom: 15px; 
                    display: flex; flex-direction: column; align-items: center;">
                    <div style="color: #00CCFF; margin-bottom: 10px; opacity: 0.8;">
                    <i class="bi bi-camera-video-off" style="font-size: 3rem;"></i></div>
                    <h5 style="color: #FFFFFF; margin: 0; font-weight: 600;">
                    Optical Sensor Offline</h5></div>""",
                    unsafe_allow_html=True)

                if st.button("INITIALIZE CAMERA STREAM", use_container_width=True, key="btn_init_cam_clf"):
                    st.session_state.camera_enabled = True
                    st.rerun()
            else:
                cam_img = st.camera_input("Optical Sensor Active", key="camera_capture_clf",
                                          label_visibility="collapsed", on_change=camera_input_callback)
                st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
                if st.button("TERMINATE FEED", use_container_width=True, key="btn_term_cam_clf"):
                    st.session_state.camera_enabled = False
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">{ICON_SETTINGS} <span>Configuration</span></div>',
                    unsafe_allow_html=True)
        top_k_slider = st.slider("Top-K Predictions (Mocked)", 1, 10, 5, 1)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">{ICON_PRIVACY} <span>Privacy Mode</span></div>',
                    unsafe_allow_html=True)
        blur_option = st.radio("Select Privacy/Blurring Mode:", options=["None", "Blur Faces Only"], index=0,
                               help="Choose 'Blur Faces Only' to automatically anonymize people.")
        st.session_state.blur_mode = blur_option
        if blur_option != "None":
            st.session_state.blur_intensity = st.slider("Blur Intensity", 1, 100, 30, help="Control blur strength.")

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

        predict_btn = st.button("Classify Image", disabled=disable_btn)

        if predict_btn:
            if is_batch:
                st.session_state.batch_files_clf = uploaded_files
                st.session_state.loading = True
                st.session_state.results_ready = False
                st.session_state.batch_results_clf = []
                st.rerun()
            else:
                st.session_state.loading = True
                st.session_state.results_ready = False
                st.session_state.classification_result = {}
                st.session_state.error = None
                st.session_state.processed_image_bytes = None
                st.session_state.batch_results_clf = None
                st.rerun()

    with input_frame:
        scan_class = "scan-effect" if st.session_state.loading else (
            "completed-effect" if st.session_state.results_ready else "")

        if uploaded_files and len(uploaded_files) > 1 and len(uploaded_files) <= 5:
            st.markdown(f"""
                <div style="height: 445px; display: flex; flex-direction: column; 
                justify-content: center; align-items: center; 
                border: 2px dashed rgba(0, 204, 255, 0.5); border-radius: 10px; 
                background: rgba(0, 12, 26, 0.8);">
                    <div style="color: #00CCFF; margin-bottom: 15px;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" 
                        viewBox="0 0 24 24" fill="none" stroke="currentColor" 
                        stroke-width="2" stroke-linecap="round" stroke-linejoin="round" 
                        class="feather feather-layers">
                        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                        <polyline points="2 17 12 22 22 17"></polyline>
                        <polyline points="2 12 12 17 22 12"></polyline>
                        </svg>
                    </div>
                    <h3 style="color: #FFFFFF; margin: 0;">Batch Mode Active</h3>
                    <p style="color: #AAAAAA; font-size: 1.1rem; margin: 5px 0;">
                    {len(uploaded_files)} Images Selected</p>
                    <small style="color: #00CCFF;">Ready to process queue</small>
                </div>
                """, unsafe_allow_html=True)

        elif st.session_state.original_image_bytes:
            img_b64 = base64.b64encode(st.session_state.original_image_bytes).decode()
            st.markdown(
                f"""<div class="img-container {scan_class}">
                    <img src="data:image/png;base64,{img_b64}" />
                </div>""",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"""<div style="height: 445px; display: flex; flex-direction: column; 
                justify-content: center; align-items: center; 
                border: 2px dashed rgba(0, 204, 255, 0.3); border-radius: 10px; 
                background: rgba(0, 0, 0, 0.2);">
                <div style="color: #00CCFF; margin-bottom: 15px;">{ICON_IMAGE}</div>
                <p style="color: #AAAAAA; font-size: 1rem; margin: 0;">Waiting for image...</p>
                <p style="color: #00CCFF; font-size: 0.9rem; margin-top: 5px;">
                Limit 200MB • JPG, PNG, BMP</p></div>""",
                unsafe_allow_html=True)

    with spacer:
        status_placeholder = st.empty()
        if st.session_state.loading:
            status_placeholder.markdown(
                """<div class="status-icon-container">
                <div class="transfer-arrow"><span></span><span></span><span></span></div>
                </div>""",
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

        result_frame = st.container(border=True, height=FRAME_HEIGHT)

        if st.session_state.loading and st.session_state.batch_files_clf:
            with result_frame:
                progress_bar = st.progress(0)
                status_text = st.empty()
                batch_history_temp = []
                files_to_process = st.session_state.batch_files_clf

                for i, file in enumerate(files_to_process):
                    status_text.markdown(
                        f"""<div style="text-align:center; padding:20px;">
                        <p style="color:#00CCFF; font-weight:bold;">
                        Processing Image {i + 1}/{len(files_to_process)}</p>
                        <small>{file.name}</small></div>""",
                        unsafe_allow_html=True)

                    try:
                        file.seek(0)
                        file_bytes = file.getvalue()
                        img_for_processing = Image.open(io.BytesIO(file_bytes))

                        start_time = time.time()
                        result = classify_image(img_for_processing)
                        end_time = time.time()
                        inf_time = end_time - start_time

                        top_conf = result['confidence']
                        df_data = [{'Class': result['class'], 'Confidence': top_conf}]
                        remaining = 1.0 - top_conf
                        other_classes = [c for c in CLASS_NAMES if c != result['class']]
                        for _ in range(top_k_slider - 1):
                            if not other_classes:
                                break
                            c = np.random.choice(other_classes)
                            other_classes.remove(c)
                            val = np.random.uniform(0, remaining)
                            remaining -= val
                            df_data.append({'Class': c, 'Confidence': val})
                        chart_data = pd.DataFrame(df_data)

                        processed_img_bytes = None
                        if st.session_state.blur_mode == "Blur Faces Only":
                            blurred_img, faces_found = apply_smart_face_blur(
                                img_for_processing,
                                st.session_state.blur_intensity
                            )
                            if faces_found:
                                buf = io.BytesIO()
                                blurred_img.save(buf, format="PNG")
                                processed_img_bytes = buf.getvalue()

                        history_item = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "original": file_bytes,
                            "processed": processed_img_bytes,
                            "class_result": result,
                            "chart_data": chart_data,
                            "inference_time": inf_time,
                            "blur_mode": st.session_state.blur_mode,
                            "blur_intensity": st.session_state.blur_intensity
                        }
                        st.session_state.history.append(history_item)
                        batch_history_temp.append(history_item)

                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")

                    progress_bar.progress((i + 1) / len(files_to_process))

                st.session_state.batch_results_clf = batch_history_temp
                st.session_state.batch_files_clf = []
                st.session_state.loading = False
                st.rerun()

        with result_frame:
            if st.session_state.loading and not st.session_state.batch_files_clf:
                st.markdown(
                    """<div class="custom-loader"><div class="loader-spinner"></div>
                    <p>Analyzing Image via Azure...</p>
                    <small>This may take a moment.</small></div>""",
                    unsafe_allow_html=True)

            elif st.session_state.batch_results_clf:
                st.markdown(
                    """<div style="height:100%; display:flex; flex-direction:column; 
                    justify-content:center; align-items:center; text-align:center;">
                    <div style="color:#00CCFF; font-size:3rem; margin-bottom:10px;">
                    <i class="bi bi-check-circle-fill"></i></div>
                    <h3 style="color:white;">Batch Complete!</h3>
                    <p style="color:#aaa;">Processed images are saved in history.</p></div>""",
                    unsafe_allow_html=True)

            elif st.session_state.results_ready:
                if st.session_state.error:
                    show_custom_toast(f"Error: {st.session_state.error}", "error")

                elif st.session_state.classification_result:

                    if st.session_state.processed_image_bytes and st.session_state.blur_mode != "None":
                        img_b64_res = base64.b64encode(st.session_state.processed_image_bytes).decode()
                        st.markdown(
                            f"""<div class="img-container completed-effect">
                            <img src="data:image/png;base64,{img_b64_res}" />
                            <div style="position: absolute; bottom: 15px; left: 50%; 
                            transform: translateX(-50%); background: rgba(0, 12, 30, 0.8); 
                            color: #00CCFF; padding: 4px 12px; border-radius: 12px; 
                            font-size: 0.75rem; border: 1px solid rgba(0, 204, 255, 0.5); 
                            display: flex; align-items: center; gap: 5px;">
                            {ICON_PRIVACY.replace('width="24"', 'width="14"').replace('height="24"', 'height="14"')}
                            <span>Privacy Protected</span></div></div>""",
                            unsafe_allow_html=True)
                    else:
                        img_b64_orig = base64.b64encode(st.session_state.original_image_bytes).decode()
                        st.markdown(
                            f"""<div class="img-container completed-effect">
                            <img src="data:image/png;base64,{img_b64_orig}" />
                            </div>""",
                            unsafe_allow_html=True)

            else:
                st.markdown(
                    f"""<div style="height: 100%; min-height: 400px; display: flex; 
                    flex-direction: column; justify-content: center; 
                    align-items: center; text-align: center; color: #777;">
                    <div style="margin-bottom: 5px; opacity: 0.6;">
                    {ICON_RESULTS_PLACEHOLDER.replace('width="64"', 'width="64"')}</div>
                    <p style="margin: 0; margin-bottom: 10px; font-size: 1.1rem; font-weight: 500;">
                    Results will appear here</p>
                    <small style="opacity: 0.8;margin-bottom: 60px;">
                    Upload an image and click Classify.</small></div>""",
                    unsafe_allow_html=True)

        if st.session_state.results_ready and not st.session_state.error and not st.session_state.batch_results_clf:
            prediction = st.session_state.classification_result
            st.markdown(
                f"""<div class="top-prediction-card"><h4>Identified Class</h4>
                <h2 class="class-name">{prediction['class'].title()}</h2>
                <h1 class="confidence-value">{prediction['confidence']:.1%}</h1>
                <p style="color: #AAAAAA; font-size: 0.8rem; margin-top: 0.5rem;">
                Confidence Score</p></div>""",
                unsafe_allow_html=True)

            if not st.session_state.chart_data.empty:
                st.markdown(
                    f"""<div style="display: flex; align-items: center; gap: 10px; 
                    margin-top: 20px; margin-bottom: 15px;">
                    <div style="color: #00CCFF;">{ICON_LIST}</div>
                    <h3 style="margin: 0; color: #E0E0E0; font-size: 1.2rem;">
                    Probability Breakdown</h3></div>""",
                    unsafe_allow_html=True)

                for idx, row in st.session_state.chart_data.iterrows():
                    rank = idx + 1
                    width_pct = row['Confidence'] * 100
                    class_name = row['Class'].title()
                    confidence_txt = f"{row['Confidence']:.1%}"
                    opacity = 0.3 + (row['Confidence'] * 0.7)

                    st.markdown(
                        f"""<div class="prob-card"><div class="prob-header">
                        <div class="prob-class"><span class="prob-rank">#{rank}</span>
                        {class_name}</div><div class="prob-score">{confidence_txt}</div></div>
                        <div class="prob-bar-bg"><div class="prob-bar-fill" 
                        style="width: {width_pct}%; opacity: {opacity};"></div></div></div>""",
                        unsafe_allow_html=True)

    if st.session_state.loading and not st.session_state.batch_files_clf:
        try:
            if not st.session_state.original_image_bytes:
                st.session_state.error = "No image found.";
                st.session_state.loading = False;
                st.session_state.results_ready = True;
                st.rerun();
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
                c = np.random.choice(other_classes);
                other_classes.remove(c);
                val = np.random.uniform(0, remaining);
                remaining -= val;
                df_data.append({'Class': c, 'Confidence': val})
            st.session_state.chart_data = pd.DataFrame(df_data)
            processed_img_bytes = None
            if st.session_state.blur_mode == "Blur Faces Only":
                blurred_img, faces_found = apply_smart_face_blur(img_for_processing, st.session_state.blur_intensity)
                if faces_found:
                    buf = io.BytesIO();
                    blurred_img.save(buf, format="PNG");
                    processed_img_bytes = buf.getvalue();
                    st.session_state.processed_image_bytes = processed_img_bytes
                else:
                    show_custom_toast("No faces detected for Privacy Mode.", "warning");
                    st.session_state.processed_image_bytes = None
            else:
                st.session_state.processed_image_bytes = None
            history_item = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "original": st.session_state.original_image_bytes,
                "processed": st.session_state.processed_image_bytes,
                "class_result": st.session_state.classification_result,
                "chart_data": st.session_state.chart_data,
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

    if st.session_state.batch_results_clf:
        zip_buffer = io.BytesIO()
        timestamp_batch = datetime.now().strftime("%Y%m%d_%H%M%S")
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for idx, item in enumerate(st.session_state.batch_results_clf):
                res_class = item['class_result']['class']
                img_data = item.get("processed") if item.get("processed") else item["original"]
                file_name = f"{idx + 1}_{res_class}_{item['timestamp'].replace(':', '-')}.png"
                zip_file.writestr(file_name, img_data)
        st.download_button(label="Download Batch Results (ZIP)", data=zip_buffer.getvalue(),
                           file_name=f"clf_batch_results_{timestamp_batch}.zip", mime="application/zip",
                           use_container_width=True, key="dl_batch_zip_clf")
        if st.button("Clear Batch Results", use_container_width=True):
            st.session_state.batch_results_clf = None
            st.rerun()

    if st.session_state.results_ready and not st.session_state.error and not st.session_state.chart_data.empty and not st.session_state.batch_results_clf:
        st.markdown("---")
        st.markdown(
            f"""<div class="section-header" style="border-color: #00CCFF; margin-top: 1.5rem;"> {ICON_STATS}<span>Classification Statistics</span></div>""",
            unsafe_allow_html=True)
        df = st.session_state.chart_data
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_bar = px.bar(df, x='Class', y='Confidence', title="Confidence Distribution", color='Confidence',
                             color_continuous_scale='Blues', text_auto='.1%')
            fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_chart2:
            fig_pie = px.pie(df, names='Class', values='Confidence', title="Probability Ratio",
                             color_discrete_sequence=px.colors.sequential.Teal)
            fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            fig_pie.update_traces(textposition='inside', textinfo='label+percent+value')
            st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("---")
        st.markdown(
            """<style>[data-testid="stDataFrame"] { background: rgba(0, 0, 0, 0.2); border: 1px solid rgba(0, 204, 255, 0.3); border-radius: 10px; } [data-testid="stDataFrame"] th { background-color: rgba(0, 204, 255, 0.15) !important; border-bottom: 2px solid #00CCFF !important; color: white !important; } </style>""",
            unsafe_allow_html=True)
        st.markdown("### Detailed Probabilities")
        display_df = df.copy()
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2%}")
        st.dataframe(display_df, use_container_width=True)
        st.markdown("---")
        st.markdown("### PDF Report")
        if st.button("Generate Full PDF Report", use_container_width=True, key="pdf_report_btn_main"):
            with st.spinner("Generating PDF report..."):
                pdf_data, filename = generate_full_classification_report()
                if pdf_data:
                    show_custom_toast("PDF Generated Successfully!", "success")
                    st.download_button(label="Download PDF Report", data=pdf_data, file_name=filename,
                                       mime="application/pdf", use_container_width=True)
                else:
                    show_custom_toast("Failed to generate PDF.", "error")

    st.markdown("---")

    st.markdown(
        """<style>.summary-metric-card { background-color: rgba(0, 0, 0, 0.2); border: 1px solid rgba(0, 204, 255, 0.3); border-radius: 10px; padding: 20px 10px; text-align: center; transition: all 0.3s ease-in-out; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; } .summary-metric-card:hover { border-color: #00CCFF; box-shadow: 0 0 15px rgba(0, 204, 255, 0.3); transform: translateY(-5px); background-color: rgba(0, 204, 255, 0.05); } .summary-metric-card .label { color: #AAAAAA; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; } .summary-metric-card .value { color: #FFFFFF; font-size: 2rem; font-weight: 700; text-shadow: 0 0 10px rgba(0, 204, 255, 0.5); line-height: 1.2; } .summary-metric-card .sub-text { font-size: 0.85rem; margin-top: 5px; font-weight: 500; } .positive { color: #00CCFF; } .negative { color: #FF4136; } .neutral { color: #777; } </style>""",
        unsafe_allow_html=True)
    metric_time_val, metric_time_txt, metric_time_cls = "N/A", "Ready", "neutral"
    metric_top1_val, metric_top1_txt, metric_top1_cls = "N/A", "No Data", "neutral"
    metric_status_val, metric_status_txt, metric_status_cls = "Idle", "Waiting", "neutral"

    if st.session_state.results_ready and not st.session_state.error and not st.session_state.batch_results_clf:
        inf_time = st.session_state.inference_time
        metric_time_val = f"{inf_time:.2f}s"
        if inf_time < 1.0:
            metric_time_txt = "Fast"; metric_time_cls = "positive"
        else:
            metric_time_txt = "Normal"; metric_time_cls = "negative"
        if st.session_state.classification_result:
            top_conf = st.session_state.classification_result['confidence']
            metric_top1_val = f"{top_conf:.1%}"
            if top_conf >= 0.8:
                metric_top1_txt = "High Confidence"; metric_top1_cls = "positive"
            else:
                metric_top1_txt = "Low Confidence"; metric_top1_cls = "negative"
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

    render_history_section()
    render_footer()


if __name__ == "__main__":
    main()