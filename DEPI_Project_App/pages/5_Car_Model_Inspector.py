import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import io
import time
import base64
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

try:
    from api_client_car import call_car_model
except ImportError:
    def call_car_model(image):
        time.sleep(1.5)
        return {
            "predictions": {
                "make": {"class_name": "Toyota", "confidence": 0.98},
                "model": {"class_name": "Corolla", "confidence": 0.92},
                "year": {"class_name": "2018-2021", "confidence": 0.89},
                "color": {"class_name": "Silver", "confidence": 0.90}
            },
            "success": True
        }

try:
    from navbar.navbar import render_navbar
    from footer.footer import render_footer
except ImportError:
    def render_navbar():
        pass

    def render_footer():
        pass

ICON_CAR = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#00CCFF" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2"></path><circle cx="7" cy="17" r="2"></circle><circle cx="17" cy="17" r="2"></circle><path d="M5 17h2"></path><path d="M15 17h2"></path></svg>"""
ICON_UPLOAD = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 16 12 12 8 16"></polyline><line x1="12" y1="12" x2="12" y2="21"></line><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path><polyline points="16 16 12 12 8 16"></polyline></svg>"""
ICON_RESULTS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>"""
ICON_HISTORY = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>"""
ICON_IMAGE = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>"""

car_icon_path = "car_icon_blue.svg"
with open(car_icon_path, "w") as f:
    f.write(ICON_CAR.replace("currentColor", "#00CCFF"))

st.set_page_config(
    page_title="Smart Car Inspector - DEPI",
    page_icon=car_icon_path,
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
    @import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css");

    html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #020c1a; border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 204, 255, 0.5); border-radius: 4px; border: 1px solid #020c1a; }
    ::-webkit-scrollbar-thumb:hover { background: #00CCFF; box-shadow: 0 0 10px rgba(0, 204, 255, 0.7); }

    .body-bg {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -2;
        background: linear-gradient(-45deg, #020c1a, #0b2f4f, #005f73, #0a9396);
        background-size: 400% 400%; animation: gradientBG 20s ease infinite;
    }
    @keyframes gradientBG { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

    .main-header-container { display: flex; align-items: center; gap: 20px; margin-bottom: 1rem; margin-top: -50px; }
    .main-header-container .icon-box { display: flex; justify-content: center; align-items: center; color: #00CCFF; text-shadow: 0 0 15px rgba(0, 204, 255, 0.5); }
    .main-header-container .text-box h1 { font-size: 2.75rem; font-weight: 700; margin: 0; line-height: 1.1; background: linear-gradient(90deg, #33DFFF, #00CCFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .main-header-container .text-box p { font-size: 1.1rem; font-weight: 300; color: #BBBBBB; margin: 0.5rem 0 0 0; }
    .styled-hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 204, 255, 0), rgba(0, 204, 255, 0.5), rgba(0, 204, 255, 0)); margin-top: 0.5rem; margin-bottom: 1.5rem; }
    .section-header { border-bottom: 2px solid #00CCFF; padding-bottom: 10px; margin-bottom: 1.5rem; font-weight: 600; font-size: 1.5rem; display: flex; align-items: center; gap: 12px; }
    .section-header svg { color: #00CCFF; }

    .stButton > button { width: 100%; background-color: #00CCFF; color: #020c1a; font-weight: 700; border: none; padding: 0.75rem; transition: all 0.3s ease; }
    .stButton > button:hover { background-color: #33DFFF; transform: translateY(-2px); box-shadow: 0 0 15px rgba(0, 204, 255, 0.4); }
    .stButton > button:disabled { background-color: rgba(255, 255, 255, 0.2); color: #888888; }
    .browse-button-only { margin-top: 0.5rem; }
    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        background-color: rgba(0, 204, 255, 0.03) !important; border: 2px dashed rgba(0, 204, 255, 0.5) !important; border-radius: 10px !important; padding: 1rem; transition: border 0.3s;
    }
    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #00CCFF !important; background-color: rgba(0, 204, 255, 0.1) !important;
    }

    .img-container { width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; padding: 20px; position: relative; border-radius: 8px; overflow: hidden; }
    .img-container img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; }
    .img-container.scan-effect { border: 2px solid rgba(0, 204, 255, 0.5); animation: pulse-border 1.5s infinite alternate; }
    .img-container.scan-effect::before { 
        content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
        background-image: linear-gradient(#00CCFF 1px, transparent 1px), linear-gradient(90deg, #00CCFF 1px, transparent 1px);
        background-size: 40px 40px; background-position: 0 0; z-index: 10; opacity: 0.3;
        animation: grid-move 3s linear infinite;
    }
    @keyframes grid-move { 0% { background-position: 0 0; } 100% { background-position: 40px 40px; } }
    @keyframes pulse-border { from { box-shadow: 0 0 5px rgba(0, 204, 255, 0.3); } to { box-shadow: 0 0 20px rgba(0, 204, 255, 0.6); } }

    .custom-loader { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; padding-top: 150px; }
    .custom-loader .loader-spinner { width: 50px; height: 50px; border: 4px solid rgba(255, 255, 255, 0.2); border-top-color: #00CCFF; border-radius: 50%; animation: spin 1s linear infinite; }
    .custom-loader p { font-weight: 600; color: #E0E0E0; animation: pulse-text 1.5s infinite ease-in-out; margin: 10px 0 0 0; }
    @keyframes pulse-text { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    .summary-metric-card { background-color: rgba(0, 0, 0, 0.2); border: 1px solid rgba(0, 204, 255, 0.3); border-radius: 10px; padding: 20px 10px; text-align: center; transition: all 0.3s ease-in-out; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; }
    .summary-metric-card:hover { border-color: #00CCFF; box-shadow: 0 0 15px rgba(0, 204, 255, 0.3); transform: translateY(-5px); background-color: rgba(0, 204, 255, 0.05); }
    .summary-metric-card .label { color: #AAAAAA; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .summary-metric-card .value { color: #FFFFFF; font-size: 1.3rem; font-weight: 700; text-shadow: 0 0 10px rgba(0, 204, 255, 0.5); line-height: 1.2; word-wrap: break-word;}
    .summary-metric-card .sub-text { font-size: 0.75rem; margin-top: 5px; font-weight: 500; }
    .positive { color: #00CCFF; } .neutral { color: #777; }

    .history-card-container { background-color: rgba(2, 12, 26, 0.6); border: 1px solid rgba(0, 204, 255, 0.2); border-left: 3px solid #00CCFF; padding: 15px; border-radius: 0 8px 8px 0; transition: all 0.3s ease; }
    .history-card-container:hover { background-color: rgba(0, 204, 255, 0.05); border-color: #00CCFF; transform: translateX(5px); }
    .hist-title { font-family: 'Poppins', sans-serif; font-weight: 600; color: #FFFFFF; font-size: 1.1rem; margin: 0; }
    .hist-meta { font-family: 'Courier New', monospace; font-size: 0.85rem; color: #AAAAAA; margin-top: 4px; }
    .hist-badge { background-color: rgba(0, 204, 255, 0.15); color: #00CCFF; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; border: 1px solid rgba(0, 204, 255, 0.3); }
    .no-data-box { text-align: center; padding: 40px; border: 1px dashed rgba(255,255,255,0.2); border-radius: 8px; color: #666; }

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
    <div style="position: fixed; top: 80px; right: 20px; z-index: 99999; width: 350px; animation: toastIn 0.5s forwards;">
        <div style="background: rgba(2, 12, 26, 0.95); backdrop-filter: blur(10px); border-left: 5px solid {'#FF4136' if type == 'error' else '#00CCFF'}; border-radius: 4px; padding: 15px 20px; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5); color: #fff; display: flex; align-items: center; gap: 15px;">
            <div style="font-size: 1.5rem; color: {'#FF4136' if type == 'error' else '#00CCFF'};"><i class="bi bi-{icon}"></i></div>
            <div>
                <h4 style="margin: 0 0 5px 0; font-size: 1rem; font-weight: 700; color: #fff;">{title}</h4>
                <p style="margin: 0; font-size: 0.85rem; color: #ddd;">{message}</p>
            </div>
        </div>
    </div>
    <style>@keyframes toastIn {{ from {{ opacity: 0; transform: translateX(100%); }} to {{ opacity: 1; transform: translateX(0); }} }}</style>
    """
    st.markdown(html_code, unsafe_allow_html=True)


def generate_car_report(history_item):
    try:
        original_bytes = history_item['original']
        preds = history_item['predictions']
        inf_time = history_item.get('inference_time', 0.0)

        try:
            dt_obj = datetime.strptime(history_item.get('timestamp', ''), "%Y-%m-%d %H:%M:%S")
            natural_date = dt_obj.strftime("%Y")
        except:
            natural_date = datetime.now().strftime("%Y")

        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"DEPI_Car_Report_{timestamp_file}.pdf"
        buffer = io.BytesIO()

        PAGE_W, PAGE_H = A4
        MARGIN_X = 0.6 * inch

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
            sub_title = "Car Inspector"

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
            canvas.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 55, "VEHICLE ANALYSIS REPORT")

            canvas.setStrokeColor(COLOR_TEAL)
            canvas.line(MARGIN_X, 55, PAGE_W - MARGIN_X, 55)

            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(COLOR_DIM)
            canvas.drawString(MARGIN_X, 40, f"Generated: {natural_date}")
            canvas.drawRightString(PAGE_W - MARGIN_X, 40, f"Page {doc.page}")
            canvas.restoreState()

        styles = getSampleStyleSheet()
        style_h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=15,
                                  textColor=COLOR_NEON, spaceBefore=20, spaceAfter=12)

        story = []
        story.append(Spacer(1, 0.25 * inch))

        story.append(Paragraph("01 // EXECUTIVE SUMMARY", style_h1))

        make = preds.get('make', {})
        model = preds.get('model', {})
        year = preds.get('year', {})

        col_w = (PAGE_W - 2 * MARGIN_X) / 3
        kpi_data = [
            ["DETECTED MAKE", "DETECTED MODEL", "EST. YEAR"],
            [make.get('class_name', 'Unknown'), model.get('class_name', 'Unknown'), year.get('class_name', 'Unknown')]
        ]

        kpi_table = Table(kpi_data, colWidths=[col_w, col_w, col_w])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLOR_PANEL),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLOR_DIM),
            ('TEXTCOLOR', (0, 1), (-1, 1), COLOR_NEON),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, 1), 14),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 12),
            ('BOX', (0, 0), (-1, -1), 0.4, COLOR_TEAL)
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 22))

        story.append(Paragraph("02 // VISUAL EVIDENCE", style_h1))
        if original_bytes:
            img_io = io.BytesIO(original_bytes)
            pil_img = Image.open(img_io).convert("RGB")
            orig_w, orig_h = pil_img.size
            MAX_W = 400
            MAX_H = 250
            scale = min(MAX_W / orig_w, MAX_H / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)

            img_buffer = io.BytesIO()
            pil_img.resize((new_w, new_h)).save(img_buffer, format="JPEG")
            img_buffer.seek(0)

            rl_img = RLImage(img_buffer, width=new_w, height=new_h)
            story.append(rl_img)
            story.append(Spacer(1, 25))

        story.append(Paragraph("03 // DETAILED CONFIDENCE METRICS", style_h1))

        detail_data = [
            ["ATTRIBUTE", "PREDICTION", "CONFIDENCE SCORE"],
            ["Make / Brand", make.get('class_name', '-'), f"{make.get('confidence', 0):.2%}"],
            ["Vehicle Model", model.get('class_name', '-'), f"{model.get('confidence', 0):.2%}"],
            ["Manufacturing Year", year.get('class_name', '-'), f"{year.get('confidence', 0):.2%}"]
        ]

        col_widths = [PAGE_W * 0.3, PAGE_W * 0.35, PAGE_W * 0.2]
        det_table = Table(detail_data, colWidths=col_widths, repeatRows=1)
        det_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0, 1, 1, 0.15)),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLOR_NEON),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 1), (-1, -1), COLOR_TEXT),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LINEBELOW', (0, 0), (-1, -1), 0.3, colors.Color(1, 1, 1, 0.1))
        ]))
        story.append(det_table)

        doc.build(story, onFirstPage=header_footer_gen, onLaterPages=header_footer_gen)
        return buffer.getvalue(), filename

    except Exception as e:
        return None, None


def main():
    st.markdown('<div class="body-bg"></div>', unsafe_allow_html=True)
    render_navbar()

    if 'history' not in st.session_state: st.session_state.history = []
    if 'car_loading' not in st.session_state: st.session_state.car_loading = False
    if 'car_result' not in st.session_state: st.session_state.car_result = None
    if 'uploaded_car_bytes' not in st.session_state: st.session_state.uploaded_car_bytes = None

    st.markdown(f"""
        <div class="main-header-container">
            <div class="icon-box" style="animation: pulse-tech 3s infinite;">{ICON_CAR}</div>
            <div class="text-box">
                <h1>Smart Car Inspector</h1>
                <p>AI-powered vehicle analysis identifying Make, Model, and Year.</p>
            </div>
        </div>
        <hr class="styled-hr">
    """, unsafe_allow_html=True)

    col1, spacer, col2 = st.columns([1, 0.15, 1.2])

    with col1:
        st.markdown(f'<div class="section-header">{ICON_UPLOAD} <span>Input Vehicle</span></div>',
                    unsafe_allow_html=True)

        visual_frame = st.container(border=True, height=350)

        st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="browse-button-only">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            if st.session_state.uploaded_car_bytes != uploaded_file.getvalue():
                st.session_state.uploaded_car_bytes = uploaded_file.getvalue()
                st.session_state.car_result = None

        st.markdown("<br>", unsafe_allow_html=True)
        btn_disabled = st.session_state.car_loading or (st.session_state.uploaded_car_bytes is None)

        if st.button("Start Inspection", use_container_width=True, disabled=btn_disabled):
            st.session_state.car_loading = True
            st.rerun()

        with visual_frame:
            scan_class = "scan-effect" if st.session_state.car_loading else ""
            if st.session_state.uploaded_car_bytes:
                img_b64 = base64.b64encode(st.session_state.uploaded_car_bytes).decode()
                st.markdown(f"""
                    <div class="img-container {scan_class}">
                        <img src="data:image/png;base64,{img_b64}" />
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; color: #777;">
                        <div style="margin-bottom: 5px; opacity: 0.6;">{ICON_IMAGE}</div>
                        <p style="margin: 0; font-size: 1rem;">Waiting for vehicle image...</p>
                    </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="section-header">{ICON_RESULTS} <span>Inspection Report</span></div>',
                    unsafe_allow_html=True)

        result_container = st.container(border=False)

        with result_container:
            if st.session_state.car_loading:
                st.markdown("""
                    <div class="custom-loader" style="padding-top:50px;">
                        <div class="loader-spinner"></div>
                        <p>Identifying Vehicle Attributes...</p>
                        <small>Querying Azure Multi-Model Endpoint</small>
                    </div>
                """, unsafe_allow_html=True)

                try:
                    image_pil = Image.open(io.BytesIO(st.session_state.uploaded_car_bytes)).convert("RGB")
                    start_time = time.time()
                    result = call_car_model(image_pil)
                    end_time = time.time()

                    st.session_state.car_result = result
                    st.session_state.car_loading = False

                    if "error" not in result:
                        history_entry = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "original": st.session_state.uploaded_car_bytes,
                            "predictions": result.get("predictions", {}),
                            "inference_time": end_time - start_time
                        }
                        st.session_state.history.append(history_entry)

                    st.rerun()

                except Exception as e:
                    st.error(f"Processing Error: {e}")
                    st.session_state.car_loading = False

            elif st.session_state.car_result:
                res = st.session_state.car_result
                if "error" in res:
                    st.error(f"API Error: {res['error']}")
                else:
                    preds = res.get("predictions", {})
                    make = preds.get("make", {})
                    model = preds.get("model", {})
                    year = preds.get("year", {})

                    m1, m2, m3 = st.columns(3)

                    with m1:
                        st.markdown(f"""
                            <div class="summary-metric-card">
                                <div class="label">Make</div>
                                <div class="value" style="font-size:1.1rem;">{make.get('class_name', 'N/A')}</div>
                                <div class="sub-text positive">{make.get('confidence', 0):.1%}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                            <div class="summary-metric-card">
                                <div class="label">Model</div>
                                <div class="value" style="font-size:1.1rem;">{model.get('class_name', 'N/A')}</div>
                                <div class="sub-text positive">{model.get('confidence', 0):.1%}</div>
                            </div>
                        """, unsafe_allow_html=True)

                    with m3:
                        st.markdown(f"""
                            <div class="summary-metric-card">
                                <div class="label">Year</div>
                                <div class="value" style="font-size:1.1rem;">{year.get('class_name', 'N/A')}</div>
                                <div class="sub-text neutral">{year.get('confidence', 0):.1%}</div>
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown("### Detailed Attributes")
                    data = [
                        {"Attribute": "Make", "Prediction": make.get('class_name'),
                         "Confidence": f"{make.get('confidence', 0):.2%}"},
                        {"Attribute": "Model", "Prediction": model.get('class_name'),
                         "Confidence": f"{model.get('confidence', 0):.2%}"},
                        {"Attribute": "Year Range", "Prediction": year.get('class_name'),
                         "Confidence": f"{year.get('confidence', 0):.2%}"}
                    ]
                    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    if st.button("Generate PDF Report", use_container_width=True, key="btn_pdf_car"):
                        current_item = {
                            "original": st.session_state.uploaded_car_bytes,
                            "predictions": preds,
                            "inference_time": 0.0,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        pdf_bytes, fname = generate_car_report(current_item)
                        if pdf_bytes:
                            show_custom_toast("Report Generated Successfully", "success")
                            st.download_button("Download Report", pdf_bytes, fname, "application/pdf",
                                               use_container_width=True)
                        else:
                            st.error("Failed to generate PDF")

            else:
                st.markdown(f"""
                    <div style="height: 400px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; color: #777;">
                        <div style="margin-bottom: 5px; opacity: 0.6;">{ICON_RESULTS}</div>
                        <p style="margin: 0; margin-bottom: 10px; font-size: 1.1rem; font-weight: 500;">Analysis results will appear here</p>
                    </div>
                """, unsafe_allow_html=True)

    render_history_section()
    render_footer()


def render_history_section():
    st.markdown("---")
    st.markdown(f'<div class="section-header">{ICON_HISTORY} <span>Recent Inspections</span></div>',
                unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
            <div class="no-data-box">
                <p>No vehicle inspections recorded yet.</p>
                <small>Upload an image to start.</small>
            </div>
        """, unsafe_allow_html=True)
        return

    for i, item in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - 1 - i

        try:
            dt = datetime.strptime(item['timestamp'], "%Y-%m-%d %H:%M:%S")
            natural_date = dt.strftime("%Y")
        except:
            natural_date = item['timestamp']

        with st.container():
            c1, c2, c3 = st.columns([1, 3, 1])
            with c1:
                st.image(item['original'], use_container_width=True)
            with c2:
                preds = item['predictions']
                make = preds.get('make', {}).get('class_name', '?')
                model = preds.get('model', {}).get('class_name', '?')

                st.markdown(f"""
                    <div style="margin-left:10px;">
                        <div class="hist-title">{make} {model}</div>
                        <div class="hist-meta">{natural_date}</div>
                        <div style="margin-top:5px;">
                            <span class="hist-badge">Conf: {preds.get('make', {}).get('confidence', 0):.0%}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with c3:
                if st.button("PDF", key=f"hist_pdf_{idx}", use_container_width=True):
                    pdf_bytes, fname = generate_car_report(item)
                    if pdf_bytes:
                        st.download_button("Download", pdf_bytes, fname, "application/pdf", key=f"dl_hist_{idx}")

            st.markdown("<hr style='opacity:0.1; margin:5px 0;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()