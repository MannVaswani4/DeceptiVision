import streamlit as st
import tempfile
import os
import numpy as np
import pandas as pd
import joblib
import time

from src.feature_extractor import process_video_to_features

st.set_page_config(page_title="DeceptiVision", page_icon="👁️", layout="wide")

# --- CUSTOM CSS: CYBER-NOIR THEME & ANIMATIONS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

    /* --- RESET & BASICS --- */
    .stApp {
        background-color: #030303;
        font-family: 'Space Mono', monospace;
        color: #e0e0e0;
        overflow-x: hidden;
    }

    /* --- ANIMATED BACKGROUND GRID --- */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; 
        left: 0;
        width: 200vw; 
        height: 200vh;
        background-image: 
            linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        animation: gridMove 20s linear infinite;
        z-index: 0;
        pointer-events: none;
    }
    
    @keyframes gridMove {
        0% { transform: translate(0, 0); }
        100% { transform: translate(-20px, -20px); }
    }

    /* --- TYPOGRAPHY --- */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #fff;
        z-index: 1;
        position: relative;
    }

    /* --- GLITCH EFFECT FOR TITLE --- */
    @keyframes glitch {
        0% { text-shadow: 2px 2px #ff0055, -2px -2px #00ffff; }
        25% { text-shadow: -2px 2px #ff0055, 2px -2px #00ffff; }
        50% { text-shadow: 2px -2px #ff0055, -2px 2px #00ffff; }
        75% { text-shadow: -2px -2px #ff0055, 2px 2px #00ffff; }
        100% { text-shadow: 2px 2px #ff0055, -2px -2px #00ffff; }
    }

    .glitch-title {
        font-size: 4.5rem;
        font-weight: 900;
        text-align: center;
        color: #fff;
        position: relative;
        animation: glitch 2s infinite linear alternate-reverse;
        margin-bottom: 0.5rem;
    }

    /* --- LANDING PAGE CONTAINER --- */
    .landing-wrapper {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: radial-gradient(circle at center, rgba(10, 10, 30, 0.9) 0%, #000 90%);
        z-index: 999;
    }

    .landing-content {
        text-align: center;
        z-index: 1000;
        background: rgba(0, 0, 0, 0.6);
        padding: 3rem;
        border: 1px solid rgba(0, 255, 255, 0.2);
        box-shadow: 0 0 50px rgba(0, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }

    .typewriter {
        font-size: 1.2rem;
        color: #00ffff;
        margin-top: 1rem;
        border-right: 2px solid #00ffff;
        white-space: nowrap;
        overflow: hidden;
        animation: typing 3s steps(40, end), blink-caret .75s step-end infinite;
        display: inline-block;
    }

    @keyframes typing { from { width: 0 } to { width: 100% } }
    @keyframes blink-caret { from, to { border-color: transparent } 50% { border-color: #00ffff; } }

    /* --- BUTTON STYLING --- */
    .stButton > button {
        background: transparent;
        border: 2px solid #00ffff;
        color: #00ffff;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.2rem;
        padding: 1rem 3rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        z-index: 1001;
    }

    .stButton > button::before {
        content: "";
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.4), transparent);
        transition: 0.5s;
    }

    .stButton > button:hover {
        box-shadow: 0 0 30px #00ffff, inset 0 0 10px #00ffff;
        text-shadow: 0 0 10px #00ffff;
        background: rgba(0, 255, 255, 0.05);
        border-color: #fff;
        color: #fff;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }

    /* --- CRT SCANLINE EFFECT --- */
    .scanline {
        width: 100%;
        height: 10px;
        z-index: 9999;
        background: linear-gradient(0deg, rgba(0,0,0,0) 0%, rgba(0, 255, 255, 0.2) 50%, rgba(0,0,0,0) 100%);
        opacity: 0.1;
        position: fixed;
        bottom: 100%;
        animation: scanline 8s linear infinite;
        pointer-events: none;
    }
    @keyframes scanline {
        0% { bottom: 100%; }
        80% { bottom: 100%; }
        100% { bottom: -10%; }
    }

    /* --- DASHBOARD STYLES --- */
    .dashboard-header {
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 1px solid #333;
        padding-bottom: 1rem;
    }
    
    .prediction-box {
        text-align: center;
        padding: 2rem;
        border: 3px double;
        margin: 2rem 0;
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 0 0 10px currentColor;
    }
    .truth-mode {
        border-color: #00ffff;
        color: #00ffff;
        background: rgba(0, 255, 255, 0.05);
    }
    .lie-mode {
        border-color: #ff0055;
        color: #ff0055;
        background: rgba(255, 0, 85, 0.05);
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# --- CRT OVERLAY ---
st.markdown('<div class="scanline"></div>', unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'entered' not in st.session_state:
    st.session_state['entered'] = False

def enter_system():
    time.sleep(0.5) # Fake loading delay
    st.session_state['entered'] = True

# --- LANDING PAGE ---
if not st.session_state['entered']:
    # CSS to center the button strictly for the landing page
    st.markdown("""
    <style> 
    .stButton {
        position: fixed;
        bottom: 20%;
        left: 50%;
        transform: translateX(-50%);
        z-index: 99999;
    }
    .stButton > button {
        padding: 0.5rem 2rem;
        min-width: 200px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="landing-wrapper">
        <div class="landing-content">
            <div class="glitch-title">DECEPTI_VISION</div>
            <div style="height: 2px; background: #00ffff; width: 100%; margin: 1rem 0; box-shadow: 0 0 10px #00ffff;"></div>
            <div class="typewriter">> INITIALIZING BIOMETRIC PROTOCOLS..._</div>
            <br><br><br>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.button("ACCESS SYSTEM", on_click=enter_system)

# --- MAIN APP ---
else:
    # --- HEADER ---
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #00ffff; font-size: 3rem;">DECEPTI_VISION</h1>
        <div style="font-family: 'Space Mono'; color: #888; letter-spacing: 2px;">FORENSIC TRUTH ANALYSIS PROTOCOL V.2.0</div>
    </div>
    """, unsafe_allow_html=True)

    # --- MAIN LAYOUT ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### < SOURCE INPUT >")
        uploaded_video = st.file_uploader("UPLOAD SURVEILLANCE FOOTAGE (MP4/MOV)", type=["mp4", "mov", "avi"])

        if uploaded_video is not None:
            st.video(uploaded_video)

    with col2:
        st.markdown("### < NEURAL OUTPUT >")
        
        if uploaded_video is not None:
            # Save temp video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
                temp.write(uploaded_video.read())
                video_path = temp.name

            with st.spinner("PROCESSING BIOMETRIC SIGNALS..."):
                # Extract features
                features = process_video_to_features(
                    video_path=video_path,
                    class_label=-1,               # unknown
                    out_csv="__streamlit_temp.csv",
                    fps=2,
                    return_features=True
                )

            if features is None:
                st.error("❌ SIGNAL LOST. UNABLE TO PROCESS FRAMES.")
            else:
                # Load classifier
                clf = joblib.load("models/deception_classifier.pkl")

                X = pd.DataFrame([features], columns=clf.feature_names_in_)
                pred = clf.predict(X)[0]
                proba = clf.predict_proba(X)[0]

                label = "TRUTH" if pred == 1 else "DECEPTION"
                style_class = "truth-mode" if pred == 1 else "lie-mode"
                
                # Display Prediction
                st.markdown(f"""
                <div class="prediction-box {style_class}">
                    {label}
                </div>
                """, unsafe_allow_html=True)

                # Confidence Meters
                st.markdown("#### PROBABILITY ANALYTICS")
                
                p_lie = proba[0]
                p_truth = proba[1]
                
                st.write(f"VERACITY SCORE: **{p_truth*100:.1f}%**")
                st.progress(float(p_truth))
                
                st.write(f"DECEPTION SCORE: **{p_lie*100:.1f}%**")
                st.progress(float(p_lie))

        else:
            st.info("AWAITING VIDEO INPUT FOR ANALYSIS...")
