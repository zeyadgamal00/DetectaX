import streamlit as st
import os
import base64

try:
    from navbar.navbar import render_navbar
    from footer.footer import render_footer
except ImportError:
    st.error("Error: Could not import navbar or footer components. Make sure the app is run from the root directory.")


    def render_navbar():
        pass


    def render_footer():
        pass

ICON_BRAIN = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-cpu"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>"""

brain_icon_path = "brain_icon_blue.svg"
with open(brain_icon_path, "w") as f:
    f.write(ICON_BRAIN.replace("currentColor", "#00CCFF"))

st.set_page_config(
    page_title="DetectaX - Image Analyzer",
    page_icon=brain_icon_path,
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Error: CSS file not found at {file_path}")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(BASE_DIR, "assets", "global.css")
home_css_path = os.path.join(BASE_DIR, "Home.css")

load_css(assets_path)
load_css(home_css_path)

st.markdown("""
<style>
    /* Scrollbar */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #020c1a; border-left: 1px solid rgba(0, 204, 255, 0.1); }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #005f73, #00CCFF); border-radius: 5px; border: 2px solid #020c1a; }
    ::-webkit-scrollbar-thumb:hover { background: #00CCFF; box-shadow: 0 0 15px rgba(0, 204, 255, 0.8); }
    html { scrollbar-width: thin; scrollbar-color: #00CCFF #020c1a; }

    /* Cursor */
    body, html, button, a, div, span, input, textarea, [role="button"] {
        cursor: url('data:image/svg+xml;utf8,<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2 2L9.5 22L12.5 14L20.5 11L2 2Z" fill="%23020c1a" stroke="%2300CCFF" stroke-width="2" stroke-linejoin="round"/></svg>'), auto !important;
    }
    a:hover, button:hover, [role="button"]:hover, input:hover, select:hover {
        cursor: url('data:image/svg+xml;utf8,<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2 2L9.5 22L12.5 14L20.5 11L2 2Z" fill="%2300CCFF" stroke="%23FFFFFF" stroke-width="2" stroke-linejoin="round"/></svg>'), pointer !important;
    }

    /* Header Scanner Effect */
    .header-scanner-wrapper {
        position: relative; background: rgba(0, 12, 30, 0.7);
        border: 2px solid rgba(0, 204, 255, 0.4); border-radius: 20px;
        padding: 3rem 2rem; overflow: hidden;
        box-shadow: 0 0 40px rgba(0, 204, 255, 0.15);
        margin-top: -80px !important; margin-bottom: 80px !important;
        animation: border-pulse 3s infinite alternate;
    }
    .header-scanner-wrapper::before {
        content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        background-image: linear-gradient(rgba(0, 204, 255, 0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 204, 255, 0.15) 1px, transparent 1px);
        background-size: 50px 50px; z-index: 0;
        animation: header-grid-move 4s linear infinite;
    }
    @keyframes header-grid-move { 0% { background-position: 0 0; } 100% { background-position: 50px 50px; } }
    @keyframes border-pulse { 0% { border-color: rgba(0, 204, 255, 0.4); } 50% { border-color: rgba(0, 204, 255, 0.8); box-shadow: 0 0 20px rgba(0, 204, 255, 0.3); } 100% { border-color: rgba(0, 204, 255, 0.4); } }
    .header-content-relative { position: relative; z-index: 1; }
</style>
""", unsafe_allow_html=True)


def get_img_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None


def main():
    st.markdown('<div class="body-bg"></div>', unsafe_allow_html=True)
    render_navbar()
    render_footer()

    welcome_section_html = f"""
    <div class="header-scanner-wrapper">
    <div class="header-content-relative">
    <div class="welcome-header-container">
    <span class="welcome-icon">{ICON_BRAIN}</span>
    <div class="welcome-title">Welcome to DetectaX Image Analyzer</div>
    </div>
    <div style="text-align:center; margin-top: 1rem;">
    <p class="welcome-subtitle">This system allows you to explore the power of Artificial Intelligence in image understanding.</p>
    <p style="font-size:1.1rem; color:#CCCCCC;">You can test our advanced models for both Image Classification and Object Detection to see how AI interprets and identifies elements in pictures.</p>
    </div>
    <div class="welcome-buttons-container">
    <a href="/Image_Classification" target="_self" class="styled-button button-classify">Classify an Image</a>
    <a href="/Object_Detection" target="_self" class="styled-button button-detect">Detect Objects</a>
    </div>
    </div>
    </div>
    """
    st.markdown(welcome_section_html, unsafe_allow_html=True)

    data_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-database"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34-9-3V5"></path></svg>"""
    model_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-layers"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>"""
    enhance_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-trending-up"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>"""
    cloud_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-cloud"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"></path></svg>"""
    mlops_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-git-branch"><line x1="6" y1="3" x2="6" y2="15"></line><circle cx="18" cy="6" r="3"></circle><circle cx="6" cy="18" r="3"></circle><path d="M18 9a9 9 0 0 1-9 9"></path></svg>"""
    ui_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-layout"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>"""

    st.header("Project Objectives")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <ul class="objectives-list">
        <li>{data_icon} <span>Collect and preprocess high-quality, diverse datasets.</span></li>
        <li>{model_icon} <span>Develop and evaluate high-performance deep learning models (CNNs, YOLO).</span></li>
        <li>{enhance_icon} <span>Enhance models via transfer learning and fine-tuning.</span></li>
        <li>{cloud_icon} <span>Deploy models on Azure using containerized services and RESTful APIs.</span></li>
        <li>{mlops_icon} <span>Implement MLOps practices for tracking, versioning, and monitoring.</span></li>
        <li>{ui_icon} <span>Deliver an intuitive web interface for real-time predictions.</span></li>
    </ul>
    """, unsafe_allow_html=True)

    image_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>"""
    box_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-box"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>"""
    zap_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-zap"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>"""

    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Key Features")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <ul class="objectives-list">
        <li>{image_icon} <span><b>Image Classification:</b> Upload an image and our model will identify the primary subject.</span></li>
        <li>{box_icon} <span><b>Object Detection:</b> Our YOLO-based model identifies and draws bounding boxes around multiple objects.</span></li>
        <li>{ui_icon} <span><b>Interactive Interface:</b> A smooth, responsive UI built with Streamlit for easy interaction.</span></li>
        <li>{zap_icon} <span><b>Real-time Predictions:</b> Get instant analysis results directly in the browser.</span></li>
    </ul>
    """, unsafe_allow_html=True)

    step1_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-navigation"><polygon points="3 11 22 2 13 21 11 13 3 11"></polygon></svg>"""
    step2_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-upload-cloud"><polyline points="16 16 12 12 8 16"></polyline><line x1="12" y1="12" x2="12" y2="21"></line><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path><polyline points="16 16 12 12 8 16"></polyline></svg>"""
    step3_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-bar-chart-2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>"""

    st.markdown("<br>", unsafe_allow_html=True)
    st.header("How It Works: A Simple Guide")
    st.markdown(f"""
    <div class="how-it-works-steps">
        <div class="step-card"><span class="step-icon">{step1_icon}</span><h4>1. Choose a Service</h4><p>Select 'Classification' or 'Detection'.</p></div>
        <div class="step-card"><span class="step-icon">{step2_icon}</span><h4>2. Upload Your Image</h4><p>Use the uploader to browse your files.</p></div>
        <div class="step-card"><span class="step-icon">{step3_icon}</span><h4>3. Get Results</h4><p>The AI model will process the image.</p></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.header("Technologies Used")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <ul class="tech-list">
            <li><b>Python & Streamlit:</b> For the core logic and interactive web UI.</li>
            <li><b>TensorFlow/Keras:</b> Used for building and training the Classification models.</li>
            <li><b>PyTorch (YOLO):</b> Powering the high-performance Object Detection.</li>
            <li><b>OpenCV & PIL:</b> For all image preprocessing and manipulation tasks.</li>
            <li><b>Microsoft Azure:</b> For cloud deployment, containerization, and MLOps.</li>
        </ul>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.header("Future Roadmap")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <ul class="tech-list">
            <li><b>Model Expansion:</b> Integrate Image Segmentation and other advanced tasks.</li>
            <li><b>User Accounts:</b> Implement a system for users to save and manage their results.</li>
            <li><b>Advanced MLOps:</b> Full CI/CD pipelines for automatic model retraining.</li>
        </ul>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Meet the Team")
    st.markdown("<br>", unsafe_allow_html=True)

    linkedin_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-linkedin"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect x="2" y="9" width="4" height="12"></rect><circle cx="4" cy="4" r="2"></circle></svg>"""
    github_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-github"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>"""

    def create_team_card(name, title, image_filename, linkedin_url="#", github_url="#"):
        image_path = os.path.join(BASE_DIR, "assets", "team_images", image_filename)
        img_b64 = get_img_base64(image_path)
        if img_b64:
            img_src = f"data:image/png;base64,{img_b64}"
        else:
            img_src = f"https://api.dicebear.com/7.x/personas/svg?seed={name.replace(' ', '')}&radius=50&backgroundType=gradientLinear&backgroundColor=00ccff,0066cc"

        return f"""
        <div class="team-card">
            <div class="team-card-image"><img src="{img_src}" alt="{name}"></div>
            <div class="team-card-info">
                <h4>{name}</h4>
                <p>{title}</p>
                <div class="team-card-socials">
                    <a href="{linkedin_url}" target="_blank" class="social-icon">{linkedin_icon}</a>
                    <a href="{github_url}" target="_blank" class="social-icon">{github_icon}</a>
                </div>
            </div>
        </div>
        """

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(create_team_card(
            "Zeyad Gamal Mohamed",
            "Team Leader, Model Development & Azure Deployment",
            "Zeyad Gamal Mohamed.jpg",
            linkedin_url="https://www.linkedin.com/in/zeyad-gamal00",
            github_url="https://github.com/zeyadgamal00"
        ), unsafe_allow_html=True)

        st.markdown(create_team_card(
            "Ziad Ahmed Samir",
            "Data Collection, Preprocessing & EDA",
            "Ziad Ahmed Samir.png",
            linkedin_url="https://www.linkedin.com/in/ziad-ahmed-46591a241/",
            github_url="https://github.com/ziad-ahemd"
        ), unsafe_allow_html=True)

        st.markdown(create_team_card(
            "Basel Mohamed Mostafa",
            "Image Classification Model & MLOps",
            "Basel Mohamed Mostafa.png",
            linkedin_url="https://www.linkedin.com/in/basel-sayed-b11534243/",
            github_url="https://github.com/BaselMohamed802"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(create_team_card(
            "Abdelrahman Kamal Elkhabery",
            "Project Documentation Lead & Transfer Learning",
            "Abdelrahman Kamal Elkhabery.png",
            linkedin_url="https://www.linkedin.com/in/abdelrahman-kamal-577b09312?utm_source=share_via&utm_content=profile&utm_medium=member_android",
            github_url="https://github.com/Abdelrahman5t"
        ), unsafe_allow_html=True)

        st.markdown(create_team_card(
            "Mohamed Hamada Farghali",
            "Object Detection Model Implementation & Evaluation",
            "Mohamed Hamada Farghali.jpg",
            linkedin_url="http://linkedin.com/in/mohamed-elfouly-14ab612a5",
            github_url="https://github.com/melfouly903"
        ), unsafe_allow_html=True)

        st.markdown(create_team_card(
            "Omar Yasser Sayed",
            "Web Interface (Frontend & API Integration)",
            "Omar Yasser Sayed.png",
            linkedin_url="https://www.linkedin.com/in/omar-yasser-software-engineer/",
            github_url="https://github.com/omaryasser3060"
        ), unsafe_allow_html=True)


if __name__ == "__main__":
    main()

