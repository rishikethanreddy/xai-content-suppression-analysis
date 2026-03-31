import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import shap
import matplotlib.pyplot as plt
import re
import plotly.graph_objects as go
from streamlit_shap import st_shap
from scipy.sparse import hstack
from feature_engineering import clean_text, extract_custom_features

import subprocess
import atexit
import socket
import sys

st.set_page_config(page_title="SafeSpeak | Toxic Content Detection", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for SaaS Website Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Global Theme Settings */
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 15% 50%, rgba(45, 20, 80, 0.4), transparent 25%),
                          radial-gradient(circle at 85% 30%, rgba(20, 60, 90, 0.4), transparent 25%);
        color: #e0e0e0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 20, 0.6) !important;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .sidebar-brand {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #b08add 0%, #6cdbef 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        text-align: center;
        letter-spacing: 1.5px;
    }
    .stRadio > div {
        gap: 15px;
    }
    .stRadio label {
        color: #d1d1d1 !important;
        font-size: 16px !important;
        font-weight: 400;
        cursor: pointer;
        padding: 5px 10px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stRadio label:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #ffffff !important;
    }
    
    /* Hero Section */
    .hero-container {
        padding: 80px 20px;
        text-align: center;
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        margin-bottom: 50px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 64px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 20px;
        background: linear-gradient(to right, #ffffff, #a5a5a5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 20px;
        color: #aeb4b8;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
        font-weight: 300;
    }
    
    /* Cards for Features */
    .feature-card {
        background: rgba(15, 15, 20, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 30px;
        text-align: left;
        height: 100%;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .feature-card:hover {
        transform: translateY(-10px) scale(1.02);
        border-color: rgba(108, 219, 239, 0.3);
        box-shadow: 0 15px 30px rgba(108, 219, 239, 0.1);
        background: rgba(25, 25, 30, 0.7);
    }
    .feature-icon {
        font-size: 36px;
        margin-bottom: 20px;
        background: rgba(255, 255, 255, 0.05);
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
    }
    .feature-title {
        color: #ffffff;
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 15px;
        letter-spacing: 0.5px;
    }
    
    /* Tool / App UI Elements */
    .stTextArea textarea {
        font-family: inherit;
        font-size: 16px;
        background: rgba(10, 10, 15, 0.7);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        transition: all 0.3s ease;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);
    }
    .stTextArea textarea:focus {
        border-color: #6cdbef;
        box-shadow: 0 0 0 2px rgba(108, 219, 239, 0.2), inset 0 2px 5px rgba(0,0,0,0.2);
        background: rgba(20, 20, 25, 0.9);
    }
    
    /* Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6cdbef 0%, #5892ff 100%);
        color: #ffffff;
        border: none;
        font-weight: 600;
        width: 100%;
        padding: 0.8rem 1.5rem;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(88, 146, 255, 0.3);
        font-size: 16px;
        letter-spacing: 0.5px;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(88, 146, 255, 0.4);
        background: linear-gradient(135deg, #7ee5f9 0%, #68a2ff 100%);
    }
    div.stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(5px);
        color: #e0e0e0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-weight: 600;
        width: 100%;
        padding: 0.8rem 1.5rem;
        border-radius: 12px;
        transition: all 0.3s ease;
        font-size: 16px;
        letter-spacing: 0.5px;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Results Visuals */
    .bar-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .bar-label {
        width: 120px;
        text-align: right;
        margin-right: 20px;
        font-size: 15px;
        font-weight: 400;
        color: #aeb4b8;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .bar-track {
        flex-grow: 1;
        background: rgba(0, 0, 0, 0.4); 
        height: 20px;
        border-radius: 10px;
        overflow: visible;
        border: 1px solid rgba(255, 255, 255, 0.05);
        position: relative;
    }
    .bar-fill {
        background: linear-gradient(90deg, #484f58, #6e7681); 
        height: 100%;
        color: #ffffff;
        text-align: right;
        padding-right: 12px;
        line-height: 20px;
        font-size: 12px;
        font-weight: 600;
        min-width: 40px;
        border-radius: 10px;
        transition: width 1s cubic-bezier(0.22, 1, 0.36, 1);
        position: absolute;
        top: 0;
        left: 0;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    /* Highlight Top Category */
    .bar-fill.highlight {
        background: linear-gradient(90deg, #bf2644, #f54266);
        box-shadow: 0 0 15px rgba(245, 66, 102, 0.5);
    }
    .bar-fill.safe {
        background: linear-gradient(90deg, #187a3e, #2ecc71);
        box-shadow: 0 0 15px rgba(46, 204, 113, 0.4);
    }
    
    .latency-text {
        text-align: right;
        color: #5d6773;
        font-size: 12px;
        margin-top: 25px;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    .main-classification {
        text-align: center;
        font-size: 64px;
        margin-bottom: 35px;
        color: #ffffff; 
        font-weight: 800;
        letter-spacing: -1.5px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .shap-header {
        font-size: 26px;
        font-weight: 600;
        color: #ffffff;
        margin-top: 60px;
        margin-bottom: 25px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding-bottom: 15px;
    }

    /* Additional Premium Tweaks */
    hr {
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    div[data-testid="stMarkdownContainer"] blockquote {
        background: rgba(108, 219, 239, 0.05);
        border-left: 4px solid #6cdbef;
        padding: 1em 1.5em;
        border-radius: 0 8px 8px 0;
        color: #c9d1d9;
        font-style: italic;
    }
    /* Tab Styling */
    button[data-baseweb="tab"] {
        font-family: inherit;
        font-weight: 600 !important;
        font-size: 16px !important;
        background-color: transparent !important;
        color: #aeb4b8 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        border-bottom-color: #6cdbef !important;
    }
    /* Upload Box Styling */
    [data-testid="stFileUploader"] {
        background: rgba(15, 15, 20, 0.5);
        border-radius: 12px;
        padding: 20px;
        border: 1px dashed rgba(255,255,255,0.1);
    }
    
    /* Make the general inputs look a bit better */
    input, .stNumberInput > div > div > input, .stSlider > div {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
DISPLAY_LABELS = {
    'toxic': 'Toxic',
    'severe_toxic': 'Severe Toxic',
    'obscene': 'Obscene',
    'threat': 'Threat',
    'insult': 'Insult',
    'identity_hate': 'Hate Speech'
}

@st.cache_resource
def load_models():
    model = joblib.load('model/xgboost_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, vectorizer, feature_names

try:
    model, vectorizer, feature_names = load_models()
except Exception as e:
    st.error("Failed to load models. Ensure `train_model.py` was run successfully.")
    st.stop()

# --- NEW: Launch suppression API Server ---
def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

if "api_process" not in st.session_state:
    if not is_port_in_use(8000):
        # Start API server in background with strict UTF-8 to prevent charmap crashes
        import os
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        proc = subprocess.Popen([sys.executable, "run_api.py"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        st.session_state.api_process = proc
        
        def cleanup_api():
            if proc.poll() is None:
                proc.terminate()
        atexit.register(cleanup_api)

# --- SIDEBAR / NAVIGATION ---
st.sidebar.markdown('<div class="sidebar-brand">SafeSpeak API</div>', unsafe_allow_html=True)
st.sidebar.markdown("<div style='color: #8b949e; text-align: center; margin-bottom: 20px;'>Content Moderation Suite</div>", unsafe_allow_html=True)

mode = st.sidebar.radio("Navigation", ["🏠 Home", "💬 Comment Analysis", "📄 Post Analysis (Docs)"])

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size: 14px; color: #8b949e;'>
<b>Technology Stack:</b><br/>
• Multi-Label Logistic Regression<br/>
• Random Forest Meta-Classifier<br/>
• TF-IDF Vectorization<br/>
• SHAP Explainable AI<br/>
• Streamlit Framework
</div>
""", unsafe_allow_html=True)

# --- PAGE: HOME ---
if mode == "🏠 Home":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Protect Your Platform. Explain Every Decision.</div>
        <div class="hero-subtitle">
            SafeSpeak is an enterprise-grade AI moderation tool designed to automatically detect toxic language, insults, obscenities, and hate speech. Unlike black-box models, we use Explainable AI (SHAP) to tell you exactly <b>why</b> content was flagged.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Platform Features", unsafe_allow_html=True)
    st.write("")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Real-Time Analysis</div>
            <div style="color: #8b949e;">Process user inputs in milliseconds. Built using highly optimized gradient boosting trees (XGBoost) combined with sparse matrix text representations.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Explainable AI (XAI)</div>
            <div style="color: #8b949e;">Every prediction is fully transparent. We utilize SHAP TreeExplainers to show your moderation team exactly which words triggered a violation policy.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Multi-Label Detection</div>
            <div style="color: #8b949e;">Content isn't just "bad". We classify text simultaneously across 6 different dimensions of toxicity, enabling granular community guidelines.</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.markdown("""
    ### 🏛️ Academic Report Statement (V4 Architecture)
    > *The upgraded system incorporates a safety-sensitive classification layer to detect crisis-related, sexual violence, and exploitation-related content, which may require distribution limitation independent of toxicity or spam indicators.*

    ### Why XAI is Required in Our System

    Our project predicts whether content will be suppressed, the reason for suppression, and the probability score. Since content suppression affects *user visibility, freedom of expression, and platform trust*, it is critical that the system does not behave like a black-box model.

    Without explainability:
    `Input text → Model → Suppressed (87%)` ... The user does not know why.

    Therefore, we integrate **SHAP (SHapley Additive exPlanations)** to ensure transparency and interpretability.

    ### 🔍 What is SHAP?
    SHAP is an Explainable AI technique based on Shapley values from cooperative game theory.
    - In game theory, each player contributes to the total outcome. The Shapley value calculates each player's fair contribution.
    - In our ML system, each word in the text is treated like a “player”. The prediction is the “game outcome”. SHAP calculates how much each word contributed to that prediction.

    ### ⚙️ How SHAP Works in Our Model
    SHAP breaks the prediction into:
    `Final Prediction = Base Value + Sum of Feature Contributions`

    **🔹 Base Value**
    The average suppression probability across the dataset (e.g., 0.40). This means if no specific text features were known, the average suppression risk is 40%.

    **🔹 Feature Contributions**
    Suppose input text: *“You are disgusting and should be banned”*
    
    SHAP might produce:
    - *disgusting* +0.22  **(Strongly increased probability)**
    - *banned* +0.15
    - *you* +0.03
    - *are* -0.01

    `Final Prediction = 0.40 + (0.22 + 0.15 + 0.03 - 0.01) ≈ 0.79`

    ### 📊 Types of Explanation Used Here
    **1️⃣ Local Explanation (Per Input)**: For every user input, SHAP identifies top contributing words, showing positive contributors that increase risk, and negative contributors that reduce risk. This ensures transparency at the individual decision level.
    **2️⃣ Global Explanation**: A summary of which features generally influence decisions across the entire platform, helping us understand algorithmic bias patterns.
    """)

# --- PAGE: COMMENT ANALYSIS ---
elif mode == "💬 Comment Analysis":
    st.markdown("## 💬 Single Comment Analysis")
    st.markdown("<p style='color: #8b949e;'>Test our moderation engine against single comments, chat messages, or short reviews.</p>", unsafe_allow_html=True)
    
    if "comment_input" not in st.session_state:
        st.session_state.comment_input = ""

    def clear_comment():
        st.session_state.comment_input = ""

    col1, spacer, col2 = st.columns([1, 0.1, 1.2])

    with col1:
        st.markdown("**INPUT TEXT**")
        comment = st.text_area("COMMENT", value=st.session_state.comment_input, height=300, placeholder="Paste a user comment here to analyze...", label_visibility="collapsed", key="text_area_input")
        
        b1, b2 = st.columns([1, 1])
        with b1:
            clear = st.button("Clear Input", on_click=clear_comment)
        with b2:
            submit = st.button("Run Prediction", type="primary")

    with col2:
        st.markdown("**ANALYSIS RESULTS**")
        
        if submit and comment.strip():
            start_time = time.time()
            import requests
            try:
                resp = requests.post("http://localhost:8000/predict-comment", json={"text": comment})
                end_time = time.time()
                if resp.status_code == 200:
                    data = resp.json()
                    dominant_label = data.get("dominant_label", "neutral")
                    probs_dict = data.get("probabilities", {})
                    
                    is_toxic = dominant_label != "neutral"
                    main_class_text = DISPLAY_LABELS.get(dominant_label, "Safe Content")
                    if dominant_label == "neutral":
                        main_class_text = "Safe Content"
                    
                    # Color logic
                    title_color = "#f85149" if is_toxic else "#3fb950" 
                    
                    st.markdown(f'<div class="main-classification" style="color: {title_color};">{main_class_text}</div>', unsafe_allow_html=True)
                    
                    max_prob = max([v for k,v in probs_dict.items() if k != 'neutral'], default=0.0) if is_toxic else probs_dict.get('neutral', 1.0)
                    
                    # Plotly Gauge Chart for overall risk
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = max_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confidence", 'font': {'size': 20, 'color': '#c9d1d9'}},
                        number = {'font': {'color': title_color, 'size': 50}, 'suffix': "%"},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#30363d"},
                            'bar': {'color': title_color},
                            'bgcolor': "#0d1117",
                            'borderwidth': 2,
                            'bordercolor': "#30363d",
                            'steps': [
                                {'range': [0, 20], 'color': "rgba(63, 185, 80, 0.2)"},
                                {'range': [20, 50], 'color': "rgba(210, 153, 34, 0.2)"},
                                {'range': [50, 100], 'color': "rgba(248, 81, 73, 0.2)"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50}
                        }
                    ))
                    fig_gauge.update_layout(paper_bgcolor="#050505", font={'color': "#e0e0e0", 'family': "Outfit"}, height=250, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # --- NEW: Explainable AI SHAP Chart ---
                    shap_data = data.get("shap_data", [])
                    if shap_data:
                        # Generate meaningful human-readable summary
                        shap_df = pd.DataFrame(shap_data)
                        top_toxic = shap_df[shap_df['value'] > 0].head(2)
                        top_safe = shap_df[shap_df['value'] < 0].tail(2)
                        
                        summary_text = f"The model determined this comment is <b>{main_class_text}</b> with {max_prob*100:.1f}% confidence. "
                        
                        if not top_toxic.empty:
                            toxic_words = ", ".join([f"'{row['feature']}'" for _, row in top_toxic.iterrows()])
                            summary_text += f"The primary signals driving this risk were the presence of the words <b>{toxic_words}</b>, which strongly increased the toxicity score. "
                            
                        if not top_safe.empty:
                            safe_words = ", ".join([f"'{row['feature']}'" for _, row in top_safe.iterrows()])
                            if top_toxic.empty:
                                summary_text += f"The content was deemed safe primarily due to the neutral context and absence of harmful terminology, with <b>{safe_words}</b> acting as stabilizing signals."
                            else:
                                summary_text += f"However, words like <b>{safe_words}</b> partially lowered the overall risk score."
                                
                        if top_toxic.empty and top_safe.empty:
                            summary_text += "The text was too brief or neutral to isolate specific keyword drivers."

                        st.markdown(f"""
                        <div style='background-color: rgba(15,15,20,0.5); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin-top: 30px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                            <div style='color: #e0e0e0; font-weight: 600; font-size: 18px; margin-bottom: 12px; letter-spacing: 0.5px;'><span style="margin-right: 8px;">💡</span> AI Behavior Summary</div>
                            <div style='color: #aeb4b8; font-size: 15px; line-height: 1.6;'>{summary_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<div class='shap-header' style='margin-top: 40px;'>🧠 Deep Model Explainability (SHAP)</div>", unsafe_allow_html=True)
                        st.markdown("<p style='color: #aeb4b8; font-size: 14px; margin-bottom: 20px;'>Understanding the precise mathematical weight of each word that drove the AI's classification decision.</p>", unsafe_allow_html=True)
                        
                        # Colors based on impact direction
                        colors = ['#f85149' if val > 0 else '#3fb950' for val in shap_df['value']]
                        
                        # Create horizontal bar chart
                        fig_shap = go.Figure(go.Bar(
                            x=shap_df['value'],
                            y=shap_df['feature'],
                            orientation='h',
                            marker_color=colors,
                            text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in shap_df['value']],
                            textposition='auto',
                            insidetextfont=dict(color='white')
                        ))
                        
                        # Style the chart
                        fig_shap.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={'color': "#e0e0e0", 'family': "Outfit"},
                            margin=dict(l=10, r=30, t=30, b=30),
                            height=max(250, len(shap_df) * 35),
                            xaxis=dict(
                                title="SHAP Value (Impact on Prediction)",
                                showgrid=True,
                                gridcolor="rgba(255,255,255,0.05)",
                                zeroline=True,
                                zerolinecolor="rgba(255,255,255,0.2)",
                                zerolinewidth=2
                            ),
                            yaxis=dict(autorange="reversed")
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                        st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 30px 0;'>", unsafe_allow_html=True)
                        
                    # --- End New Code ---
                    
                    # Bars
                    display_order = ['toxic', 'obscene', 'insult', 'very_toxic', 'hate', 'threat']
                    bars_html = ""
                    for lbl in display_order:
                        prob_pct = int(probs_dict.get(lbl, 0.0) * 100)
                        
                        # Match label names
                        disp_name = DISPLAY_LABELS.get(lbl, lbl.capitalize())
                        if lbl == 'very_toxic': disp_name = 'Severe Toxic'
                        if lbl == 'hate': disp_name = 'Hate Speech'
                        
                        css_class = "bar-fill highlight" if prob_pct >= 50 else "bar-fill"
                        bars_html += f'''
                        <div class="bar-container">
                            <div class="bar-label">{disp_name}</div>
                            <div class="bar-track">
                                <div class="{css_class}" style="width: {max(prob_pct, 1)}%;">{prob_pct}%</div>
                            </div>
                        </div>'''
                    
                    neutral_pct = int(probs_dict.get('neutral', 0.0) * 100)
                    safe_class = "bar-fill safe" if neutral_pct >= 50 else "bar-fill"
                    bars_html += f'''
                    <div class="bar-container">
                        <div class="bar-label">Neutral / Safe</div>
                        <div class="bar-track">
                            <div class="{safe_class}" style="width: {max(neutral_pct, 1)}%;">{neutral_pct}%</div>
                        </div>
                    </div>'''
                    
                    bars_html += f'<div class="latency-text">Network latency: {(end_time - start_time):.3f}s</div>'
                    st.markdown(bars_html, unsafe_allow_html=True)
                else:
                    st.error(f"API Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Failed to connect to /predict-comment API: {e}")

        else:
             st.markdown("""
             <div style="height: 300px; display: flex; align-items: center; justify-content: center; border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px; color: #aeb4b8; background: rgba(15,15,20,0.3); backdrop-filter: blur(5px);">
                Submit text to view classification scores and detailed SHAP explanations.
             </div>
             """, unsafe_allow_html=True)

# --- PAGE: POST ANALYSIS ---
elif mode == "📄 Post Analysis (Docs)":
    st.markdown("## 📊 Post Visibility Analysis")
    st.markdown("<p style='color: #8b949e;'>Analyze engagement and content signals to predict visibility suppression.</p>", unsafe_allow_html=True)
    
    tab_manual, tab_image = st.tabs(["📝 Manual Input", "🖼️ Upload Post Image"])
    
    with tab_manual:
        col1, spacer, col2 = st.columns([1, 0.1, 1.2])
        
        with col1:
            st.markdown("""
            <div style='background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 20px;'>
                <div style='color: #ffffff; font-size: 18px; font-weight: bold;'><span style='margin-right: 10px;'>📝</span>1. Content Signals</div>
            </div>
            """, unsafe_allow_html=True)
            post_text = st.text_area("Post Content", height=150, placeholder="Enter post content here...")
            
            from streamlit_tags import st_tags
            tags_input = st_tags(
                label='Hashtags',
                text='Type and press Enter, or use commas',
                value=[],
                suggestions=['#update', '#news', '#policy', '#spam'],
                maxtags=15,
                key='hashtags'
            )
            
            st.markdown("""
            <div style='background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-top: 25px; margin-bottom: 20px;'>
                <div style='color: #ffffff; font-size: 18px; font-weight: bold;'><span style='margin-right: 10px;'>📈</span>2. Engagement Signals</div>
            </div>
            """, unsafe_allow_html=True)
            likes = st.number_input("Likes", min_value=0, value=0, step=1)
            watch_time = st.slider("Watch Time %", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
            
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button("Analyze Post", type="primary", use_container_width=True)


        with col2:
            st.markdown("**VISIBILITY REPORT**")
            
            if analyze_btn:
                if not post_text.strip():
                    st.warning("Please enter post content.")
                else:
                    with st.spinner("Analyzing behavioral and content metrics..."):
                        import requests
                        
                        tags = tags_input if tags_input else []
                        
                        payload = {
                            "text": post_text,
                            "hashtags": tags,
                            "likes": int(likes),
                            "watch_time": float(watch_time)
                        }
                        
                        try:
                            resp = requests.post("http://localhost:8000/predict-post", json=payload)
                            if resp.status_code == 200:
                                data = resp.json()
                                reason = data.get("reason", "Unknown")
                                conf = data.get("confidence", 0.0)
                                explanation = data.get("explanation", "No explanation provided.")
                                
                                if reason == "Normal visibility":
                                    bg_color = "rgba(63, 185, 80, 0.1)"
                                    border_color = "#3fb950"
                                    text_color = "#3fb950"
                                    icon = "✅"
                                elif reason == "Low engagement":
                                    bg_color = "rgba(210, 153, 34, 0.1)"
                                    border_color = "#d29922"
                                    text_color = "#d29922"
                                    icon = "📉"
                                else: 
                                    bg_color = "rgba(248, 81, 73, 0.1)"
                                    border_color = "#f85149"
                                    text_color = "#f85149"
                                    icon = "🚨"
                                    
                                st.markdown(f"""
                                <div style='background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 8px; padding: 20px; text-align: center; margin-bottom: 20px;'>
                                    <div style='font-size: 48px; margin-bottom: 10px;'>{icon}</div>
                                    <div style='color: {text_color}; font-size: 24px; font-weight: bold;'>{reason}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"**Confidence Score:** {conf*100:.1f}%")
                                st.progress(conf)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                # Generate meaningful human-readable summary
                                shap_data = data.get("shap_data", [])
                                summary_text = ""
                                if shap_data:
                                    shap_df = pd.DataFrame(shap_data)
                                    top_risk = shap_df[shap_df['value'] > 0].head(2)
                                    top_safe = shap_df[shap_df['value'] < 0].tail(2)
                                    
                                    action_word = "suppressed" if reason != "Normal visibility" else "approved"
                                    summary_text = f"The AI engine <b>{action_word}</b> this post, classifying it as '{reason}' with {conf*100:.1f}% confidence. "
                                    
                                    if not top_risk.empty:
                                        risk_factors = ", ".join([f"<b>{row['feature'].lower()}</b>" for _, row in top_risk.iterrows()])
                                        summary_text += f"The decision to restrict visibility was heavily driven by high {risk_factors}. "
                                        
                                    if not top_safe.empty:
                                        safe_factors = ", ".join([f"<b>{row['feature'].lower()}</b>" for _, row in top_safe.iterrows()])
                                        if top_risk.empty:
                                            summary_text += f"The content maintained normal visibility largely due to positive engagement metrics, specifically {safe_factors}."
                                        else:
                                            summary_text += f"While metrics like {safe_factors} contributed positively, they were not enough to override the risk flags."
                                            
                                if not summary_text:
                                    summary_text = "The model evaluated the baseline metrics but found no strong diverging features to explicitly highlight."

                                st.markdown(f"""
                                <div style='background-color: rgba(15,15,20,0.5); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                                    <div style='color: #e0e0e0; font-weight: 600; font-size: 18px; margin-bottom: 12px; letter-spacing: 0.5px;'><span style="margin-right: 8px;">💡</span> AI Behavior Summary</div>
                                    <div style='color: #aeb4b8; font-size: 15px; line-height: 1.6;'>{summary_text}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # --- NEW: Post Analysis SHAP Chart ---
                                if shap_data:
                                    st.markdown("<div class='shap-header' style='margin-top: 40px;'>🧠 Deep Model Explainability (SHAP)</div>", unsafe_allow_html=True)
                                    st.markdown("<p style='color: #aeb4b8; font-size: 14px; margin-bottom: 20px;'>A mathematical breakdown of the content and engagement signals that shifted the AI's visibility decision.</p>", unsafe_allow_html=True)
                                    
                                    # Prepare data for Plotly
                                    shap_df = pd.DataFrame(shap_data)
                                    
                                    # Colors based on impact direction
                                    colors = ['#f85149' if val > 0 else '#3fb950' for val in shap_df['value']]
                                    
                                    # Create horizontal bar chart
                                    fig_shap = go.Figure(go.Bar(
                                        x=shap_df['value'],
                                        y=shap_df['feature'],
                                        orientation='h',
                                        marker_color=colors,
                                        text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in shap_df['value']],
                                        textposition='auto',
                                        insidetextfont=dict(color='white')
                                    ))
                                    
                                    # Style the chart
                                    fig_shap.update_layout(
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        font={'color': "#e0e0e0", 'family': "Outfit"},
                                        margin=dict(l=10, r=30, t=30, b=30),
                                        height=max(250, len(shap_df) * 35),
                                        xaxis=dict(
                                            title="SHAP Value (Impact on Visibility Suppression)",
                                            showgrid=True,
                                            gridcolor="rgba(255,255,255,0.05)",
                                            zeroline=True,
                                            zerolinecolor="rgba(255,255,255,0.2)",
                                            zerolinewidth=2
                                        ),
                                        yaxis=dict(autorange="reversed")
                                    )
                                    st.plotly_chart(fig_shap, use_container_width=True)
                                # --- End New Code ---
                            else:
                                st.error(f"API Error {resp.status_code}: {resp.text}")
                        except Exception as e:
                            st.error(f"Failed to connect to suppression API. Please assure the background thread is running: {e}")
            else:
                 st.markdown("""
                 <div style="height: 300px; display: flex; align-items: center; justify-content: center; border: 1px dashed #30363d; border-radius: 6px; color: #8b949e;">
                    Waiting for input data...
                 </div>
                 """, unsafe_allow_html=True)
                 
    with tab_image:
        colimg1, spacerimg, colimg2 = st.columns([1, 0.1, 1.2])
        
        with colimg1:
            st.markdown("""
            <div style='background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 20px;'>
                <div style='color: #ffffff; font-size: 18px; font-weight: bold;'><span style='margin-right: 10px;'>📸</span>Upload Screenshot</div>
                <div style='color: #8b949e; font-size: 14px; margin-top: 5px;'>Upload a screenshot of a social media post. The OCR engine will extract the caption, hashtags, likes, and views.</div>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
            
            st.markdown("<p style='text-align: center; color: #8b949e; margin: 10px 0;'>— OR —</p>", unsafe_allow_html=True)
            
            from streamlit_paste_button import paste_image_button
            paste_result = paste_image_button(
                label="📋 Paste Image from Clipboard",
                text_color="#ffffff",
                background_color="#238636",
                hover_background_color="#2ea043"
            )
            
            image_bytes = None
            image_name = None
            preview_source = None
            
            if uploaded_file is not None:
                image_bytes = uploaded_file.getvalue()
                image_name = uploaded_file.name
                preview_source = uploaded_file
            elif paste_result.image_data is not None:
                import io
                img_byte_arr = io.BytesIO()
                paste_result.image_data.save(img_byte_arr, format='PNG')
                image_bytes = img_byte_arr.getvalue()
                image_name = "pasted_image.png"
                preview_source = paste_result.image_data
                
            st.markdown("<br>", unsafe_allow_html=True)
            if preview_source is not None:
                st.image(preview_source, caption="Captured Post Screenshot", use_container_width=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_img_btn = st.button("Extract & Analyze", type="primary", use_container_width=True, disabled=image_bytes is None)
            
        with colimg2:
            st.markdown("**VISIBILITY REPORT (OCR MODE)**")
            
            if analyze_img_btn and image_bytes is not None:
                with st.spinner("Running EasyOCR extraction and multi-modal analysis... This may take a moment."):
                    import requests
                    
                    try:
                        # Force a safe ASCII filename. If the original filename contains emojis or special unicode (like \u2588)
                        # the `requests` library will crash deeply inside urllib3/httplib when trying to encode the multipart header on Windows.
                        safe_filename = "upload.png" 
                        files = {"file": (safe_filename, image_bytes, "image/png")}
                        resp = requests.post("http://localhost:8000/predict-post-image", files=files)
                        
                        if resp.status_code == 200:
                            data = resp.json()
                            reason = data.get("reason", "Unknown")
                            conf = data.get("confidence", 0.0)
                            explanation = data.get("explanation", "No explanation provided.")
                            extracted = data.get("extracted_data", {})
                            
                            # Display extracted info
                            st.markdown("### 🧩 Extracted Information")
                            st.info(f"**Caption:** {extracted.get('text', 'N/A')}")
                            st.info(f"**Hashtags:** {', '.join(extracted.get('hashtags', []))}")
                            st.info(f"**Likes:** {extracted.get('likes', 0)}")
                            st.markdown("---")
                            
                            if reason == "Normal visibility":
                                bg_color = "rgba(63, 185, 80, 0.1)"
                                border_color = "#3fb950"
                                text_color = "#3fb950"
                                icon = "✅"
                            elif reason == "Low engagement":
                                bg_color = "rgba(210, 153, 34, 0.1)"
                                border_color = "#d29922"
                                text_color = "#d29922"
                                icon = "📉"
                            else: 
                                bg_color = "rgba(248, 81, 73, 0.1)"
                                border_color = "#f85149"
                                text_color = "#f85149"
                                icon = "🚨"
                                
                            st.markdown(f"""
                            <div style='background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 8px; padding: 20px; text-align: center; margin-bottom: 20px;'>
                                <div style='font-size: 48px; margin-bottom: 10px;'>{icon}</div>
                                <div style='color: {text_color}; font-size: 24px; font-weight: bold;'>{reason}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"**Confidence Score:** {conf*100:.1f}%")
                            st.progress(conf)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Generate meaningful human-readable summary
                            shap_data = data.get("shap_data", [])
                            summary_text = ""
                            if shap_data:
                                shap_df = pd.DataFrame(shap_data)
                                top_risk = shap_df[shap_df['value'] > 0].head(2)
                                top_safe = shap_df[shap_df['value'] < 0].tail(2)
                                
                                action_word = "suppressed" if reason != "Normal visibility" else "approved"
                                summary_text = f"The AI engine <b>{action_word}</b> this image post, classifying it as '{reason}' with {conf*100:.1f}% confidence. "
                                
                                if not top_risk.empty:
                                    risk_factors = ", ".join([f"<b>{row['feature'].lower()}</b>" for _, row in top_risk.iterrows()])
                                    summary_text += f"The decision to restrict visibility was heavily driven by high {risk_factors} found in the extracted text. "
                                    
                                if not top_safe.empty:
                                    safe_factors = ", ".join([f"<b>{row['feature'].lower()}</b>" for _, row in top_safe.iterrows()])
                                    if top_risk.empty:
                                        summary_text += f"The content maintained normal visibility largely due to safe extraction metrics, specifically {safe_factors}."
                                    else:
                                        summary_text += f"While metrics like {safe_factors} contributed positively to the score, they were not enough to override the policy risk flags."
                                        
                            if not summary_text:
                                summary_text = "The multi-modal model evaluated baseline OCR metrics but found no strong diverging features to explicitly highlight."

                            st.markdown(f"""
                            <div style='background-color: rgba(15,15,20,0.5); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                                <div style='color: #e0e0e0; font-weight: 600; font-size: 18px; margin-bottom: 12px; letter-spacing: 0.5px;'><span style="margin-right: 8px;">💡</span> AI Behavior Summary</div>
                                <div style='color: #aeb4b8; font-size: 15px; line-height: 1.6;'>{summary_text}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # --- NEW: OCR Post Analysis SHAP Chart ---
                            if shap_data:
                                st.markdown("<div class='shap-header' style='margin-top: 40px;'>🧠 Deep Model Explainability (SHAP)</div>", unsafe_allow_html=True)
                                st.markdown("<p style='color: #aeb4b8; font-size: 14px; margin-bottom: 20px;'>A mathematical breakdown of the content and engagement signals that shifted the AI's visibility decision.</p>", unsafe_allow_html=True)
                                
                                # Prepare data for Plotly
                                shap_df = pd.DataFrame(shap_data)
                                
                                # Colors based on impact direction
                                colors = ['#f85149' if val > 0 else '#3fb950' for val in shap_df['value']]
                                
                                # Create horizontal bar chart
                                fig_shap = go.Figure(go.Bar(
                                    x=shap_df['value'],
                                    y=shap_df['feature'],
                                    orientation='h',
                                    marker_color=colors,
                                    text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in shap_df['value']],
                                    textposition='auto',
                                    insidetextfont=dict(color='white')
                                ))
                                
                                # Style the chart
                                fig_shap.update_layout(
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    font={'color': "#e0e0e0", 'family': "Outfit"},
                                    margin=dict(l=10, r=30, t=30, b=30),
                                    height=max(250, len(shap_df) * 35),
                                    xaxis=dict(
                                        title="SHAP Value (Impact on Visibility Suppression)",
                                        showgrid=True,
                                        gridcolor="rgba(255,255,255,0.05)",
                                        zeroline=True,
                                        zerolinecolor="rgba(255,255,255,0.2)",
                                        zerolinewidth=2
                                    ),
                                    yaxis=dict(autorange="reversed")
                                )
                                st.plotly_chart(fig_shap, use_container_width=True)
                            # --- End New Code ---
                        else:
                            st.error(f"API Error {resp.status_code}: {resp.text}")
                    except Exception as e:
                        st.error(f"Failed to connect to suppression API: {e}")
            else:
                 st.markdown("""
                 <div style="height: 300px; display: flex; align-items: center; justify-content: center; border: 1px dashed #30363d; border-radius: 6px; color: #8b949e;">
                    Waiting for image upload...
                 </div>
                 """, unsafe_allow_html=True)

