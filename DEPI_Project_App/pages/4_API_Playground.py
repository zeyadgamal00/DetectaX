import streamlit as st
import time
import json
import base64
import io
import requests
from PIL import Image
import os

try:
    from navbar.navbar import render_navbar
    from footer.footer import render_footer
except ImportError:
    def render_navbar():
        pass

    def render_footer():
        pass

try:
    from api_client import CNN_ENDPOINT, CNN_KEY, OD_ENDPOINT, OD_KEY, _pil_to_base64
except ImportError:
    CNN_ENDPOINT = ""
    CNN_KEY = ""
    OD_ENDPOINT = ""
    OD_KEY = ""

    def _pil_to_base64(image: Image.Image, format="PNG") -> str:
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

# --- New Car Model Configuration ---
CAR_ENDPOINT = ""
CAR_KEY = ""

ICON_API = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#00CCFF" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2" ry="2"></rect><line x1="12" y1="2" x2="12" y2="22"></line><line x1="2" y1="12" x2="22" y2="12"></line><circle cx="12" cy="12" r="4"></circle></svg>"""
ICON_CODE = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>"""
ICON_SERVER = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect><rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect><line x1="6" y1="6" x2="6.01" y2="6"></line><line x1="6" y1="18" x2="6.01" y2="18"></line></svg>"""
ICON_CHECK = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00CCFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>"""
ICON_UPLOAD = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-upload-cloud"><polyline points="16 16 12 12 8 16"></polyline><line x1="12" y1="12" x2="12" y2="21"></line><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path><polyline points="16 16 12 12 8 16"></polyline></svg>"""
ICON_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-settings"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>"""
ICON_TERMINAL = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-terminal"><polyline points="4 17 10 11 4 5"></polyline><line x1="12" y1="19" x2="20" y2="19"></line></svg>"""

api_icon_path = "api_icon_blue.svg"
with open(api_icon_path, "w") as f:
    f.write(ICON_API.replace("currentColor", "#00CCFF"))

st.set_page_config(
    page_title="API Integration - DetectaX",
    page_icon=api_icon_path,
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

    .stButton > button { width: 100%; background-color: #00CCFF; color: #020c1a; font-weight: 700; border: none; padding: 0.75rem; transition: all 0.3s ease; }
    .stButton > button:hover { background-color: #33DFFF; transform: translateY(-2px); box-shadow: 0 0 15px rgba(0, 204, 255, 0.4); }
    .section-header { border-bottom: 2px solid #00CCFF; padding-bottom: 10px; margin-bottom: 1.5rem; font-weight: 600; font-size: 1.5rem; display: flex; align-items: center; gap: 12px; }
    .section-header svg { color: #00CCFF; }

    .img-container { width: 100%; height: 250px; display: flex; justify-content: center; align-items: center; padding: 10px; position: relative; border-radius: 8px; overflow: hidden; background: rgba(0,0,0,0.2); border: 1px solid rgba(0,204,255,0.1); }
    .img-container img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 4px; }

    .custom-loader { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; padding: 40px; }
    .custom-loader .loader-spinner { width: 40px; height: 40px; border: 3px solid rgba(255, 255, 255, 0.2); border-top-color: #00CCFF; border-radius: 50%; animation: spin 1s linear infinite; }
    .custom-loader p { font-weight: 600; color: #E0E0E0; margin-top: 15px; font-size: 0.9rem; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    .metric-card { background: rgba(0, 12, 26, 0.6); border: 1px solid rgba(0, 204, 255, 0.2); border-radius: 8px; padding: 15px; display: flex; flex-direction: column; align-items: center; justify-content: center; transition: all 0.3s; }
    .metric-card:hover { border-color: #00CCFF; background: rgba(0, 204, 255, 0.05); }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #fff; }
    .metric-label { font-size: 0.8rem; color: #aaa; text-transform: uppercase; margin-top: 5px; }
    .status-ok { color: #00CCFF; }
    .status-err { color: #FF4136; }

    [data-testid="stJson"] { background-color: #0b1d36; border-radius: 8px; padding: 10px; border: 1px solid rgba(255,255,255,0.1); }
    .stCodeBlock { border: 1px solid rgba(0,204,255,0.2); border-radius: 8px; }

    [data-testid="stTabs"] button { color: #FFFFFF !important; font-weight: bold; }
    [data-testid="stTabs"] button[aria-selected="true"] { border-top-color: #00CCFF !important; color: #00CCFF !important; }

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

    [data-testid="stDownloadButton"] button {
        border: 1px solid rgba(0, 204, 255, 0.3) !important;
        background-color: rgba(0, 204, 255, 0.05) !important;
        color: #00CCFF !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background-color: #00CCFF !important;
        color: #000 !important;
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

def perform_api_request(endpoint, key, image, task_type):
    start_time = time.time()

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    img_b64 = _pil_to_base64(image)

    if task_type == "Object Detection":
        payload = {"image_base64": img_b64, "conf": 0.5}
    else:
        payload = {"image": img_b64}

    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
        latency = (time.time() - start_time) * 1000

        return {
            "status_code": response.status_code,
            "latency": latency,
            "json": response.json() if response.status_code == 200 else {"error": response.text},
            "headers": dict(response.headers)
        }
    except Exception as e:
        return {
            "status_code": 500,
            "latency": 0,
            "json": {"error": str(e)},
            "headers": {}
        }

def main():
    st.markdown('<div class="body-bg"></div>', unsafe_allow_html=True)
    render_navbar()

    st.markdown(f"""
        <div class="main-header-container">
            <div class="icon-box" style="animation: pulse-tech 3s infinite;">{ICON_API}</div>
            <div class="text-box">
                <h1>API Integration Hub</h1>
                <p>Test, debug, and connect with DetectaX cloud inference models seamlessly.</p>
            </div>
        </div>
        <hr class="styled-hr">
    """, unsafe_allow_html=True)

    col_config, spacer, col_result = st.columns([1, 0.05, 1.2])

    with col_config:
        st.markdown(f'<div class="section-header">{ICON_SETTINGS} <span>Request Config</span></div>',
                    unsafe_allow_html=True)

        config_container = st.container(border=True)
        with config_container:
            task_type = st.selectbox("Select Model Endpoint",
                                     ["Image Classification (CNN)", "Object Detection (YOLO)", "Car Analysis (CNN)"],
                                     help="Choose the Azure Endpoint to target.")

            if "Car" in task_type:
                active_endpoint = CAR_ENDPOINT
                active_key = CAR_KEY
                task_tag = "Classification"
            elif "Object Detection" in task_type:
                active_endpoint = OD_ENDPOINT
                active_key = OD_KEY
                task_tag = "Object Detection"
            else:
                active_endpoint = CNN_ENDPOINT
                active_key = CNN_KEY
                task_tag = "Classification"

            st.caption("Target Endpoint URL")
            st.code(active_endpoint, language="http")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="section-header">{ICON_UPLOAD} <span>Payload Input</span></div>',
                        unsafe_allow_html=True)

            uploaded_file = st.file_uploader("Upload Test Image", type=["jpg", "png", "jpeg"])

            if uploaded_file:
                image = Image.open(uploaded_file)
                st.markdown(f"""
                <div style="margin-top:10px; padding:10px; background:rgba(0,0,0,0.2); border-radius:8px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#aaa; margin-bottom:5px;">
                        <span>{image.format}</span>
                        <span>{image.size[0]}x{image.size[1]}px</span>
                        <span>{uploaded_file.size / 1024:.1f} KB</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                img_b64 = _pil_to_base64(image)
                st.markdown(f'<div class="img-container"><img src="data:image/png;base64,{img_b64}"></div>',
                            unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Execute Request", use_container_width=True):
                    st.session_state.trigger_api = True
            else:
                st.info("Upload an image to unlock the request trigger.")

    with col_result:
        st.markdown(f'<div class="section-header">{ICON_TERMINAL} <span>Response Console</span></div>',
                    unsafe_allow_html=True)

        result_container = st.container(border=True)
        with result_container:

            if st.session_state.get('trigger_api') and uploaded_file:
                with st.empty():
                    st.markdown("""
                        <div class="custom-loader">
                            <div class="loader-spinner"></div>
                            <p>Handshaking with Azure Endpoint...</p>
                        </div>
                    """, unsafe_allow_html=True)

                    result = perform_api_request(active_endpoint, active_key, image, task_tag)

                st.session_state.trigger_api = False

                status_color = "status-ok" if result['status_code'] == 200 else "status-err"

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(
                        f"""<div class="metric-card"><div class="metric-value {status_color}">{result['status_code']}</div><div class="metric-label">Status Code</div></div>""",
                        unsafe_allow_html=True)
                with m2:
                    st.markdown(
                        f"""<div class="metric-card"><div class="metric-value">{result['latency']:.0f}ms</div><div class="metric-label">Latency</div></div>""",
                        unsafe_allow_html=True)
                with m3:
                    st.markdown(
                        f"""<div class="metric-card"><div class="metric-value">POST</div><div class="metric-label">Method</div></div>""",
                        unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                tab_json, tab_py, tab_curl = st.tabs(["JSON Output", "Python Code", "cURL"])

                with tab_json:
                    json_str = json.dumps(result['json'], indent=4)
                    st.download_button(
                        label="Download JSON Response",
                        data=json_str,
                        file_name=f"api_response_{int(time.time())}.json",
                        mime="application/json",
                        use_container_width=True
                    )

                    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

                    if result['status_code'] == 200:
                        st.success("Request Successful")
                    else:
                        st.error("Request Failed")

                    st.markdown("### Response Body")
                    with st.container(height=500, border=False):
                        st.json(result['json'])

                with tab_py:
                    st.markdown("### Python Requests Integration")
                    py_code = f"""import requests
import base64
import json

# Setup
ENDPOINT = "{active_endpoint}"
API_KEY = "{active_key[:10]}..." # Truncated for security
IMAGE_PATH = "image.jpg"

# Encode
with open(IMAGE_PATH, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

# Payload
"""
                    if task_tag == "Object Detection":
                        py_code += 'payload = {"image_base64": img_b64, "conf": 0.5}'
                    else:
                        py_code += 'payload = {"image": img_b64}'

                    py_code += """

# Request
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(ENDPOINT, headers=headers, json=payload)
print(response.json())
"""
                    st.code(py_code, language="python")

                with tab_curl:
                    st.markdown("### Terminal / Bash")
                    img_str = "<BASE64_STRING>"
                    payload_str = '{"image_base64": "' + img_str + '", "conf": 0.5}' if task_tag == "Object Detection" else '{"image": "' + img_str + '"}'

                    curl_code = f"""curl -X POST "{active_endpoint}" \\
-H "Authorization: Bearer {active_key}" \\
-H "Content-Type: application/json" \\
-d '{payload_str}'"""
                    st.code(curl_code, language="bash")

            elif not uploaded_file:
                st.markdown(f"""
                    <div style="height:400px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; color:#555;">
                        <div style="opacity:0.3; margin-bottom:15px;">{ICON_SERVER}</div>
                        <h3>Awaiting Request</h3>
                        <p>Upload an image and click Execute to see live API results here.</p>
                    </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                    <div style="height:400px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; color:#555;">
                        <div style="opacity:0.5; color:#00CCFF; margin-bottom:15px; animation: pulse 2s infinite;">{ICON_CODE}</div>
                        <h3>Ready to Fire</h3>
                        <p>Click the <b>Execute Request</b> button to send data.</p>
                    </div>
                """, unsafe_allow_html=True)

    render_footer()

if __name__ == "__main__":
    main()