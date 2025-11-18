import streamlit as st
import tempfile
import os
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# === import your modules ===
from src.predict_emotion import predict_emotion
from src.body_language_yolo import extract_yolo_pose
from src.video_to_frames import extract_frames
from src.feature_extractor import process_video_to_features   # rename your function file
from src.models.face_cnn import FaceCNN
import torch


# Load classifier
clf = joblib.load("models/deception_classifier.pkl")

st.set_page_config(page_title="DeceptiVision", layout="wide")
st.title("🕵️‍♂️ DeceptiVision – Lie Detection AI")
st.write("Upload a short video and get a Truth/Lie prediction with explanation.")


# ========================
#   FILE UPLOADER
# ========================
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    st.success("Video uploaded!")

    # save to temp
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    temp_video_path = tfile.name

    st.video(temp_video_path)

    st.info("Processing video... This may take 5–20 seconds.")

    # ========================
    #   EXTRACT FEATURES
    # ========================
    try:
        features = process_video_to_features(
            video_path=temp_video_path,
            class_label=0,     # dummy
            out_csv="__temp.csv",
            fps=2
        )

        if features is None:
            st.error("Could not extract features from the video.")
            st.stop()

        # ========================
        #   FORMAT FOR MODEL
        # ========================
        X_input = pd.DataFrame([features], columns=clf.feature_names_in_)

        # Predict
        pred = clf.predict(X_input)[0]
        proba = clf.predict_proba(X_input)[0]

        result = "TRUTH" if pred == 1 else "LIE"
        color = "green" if result == "TRUTH" else "red"

        st.markdown(f"## Prediction: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)

        # Probabilities
        st.subheader("Prediction Confidence")
        st.write({
            "Lie Probability": round(float(proba[0]), 3),
            "Truth Probability": round(float(proba[1]), 3),
        })


        # ========================
        #   EXPLANATION SECTION
        # ========================
        st.subheader("Why this prediction?")

        # Top random forest feature importance
        feature_importance = clf.feature_importances_
        top_idx = np.argsort(feature_importance)[-10:][::-1]

        top_features = pd.DataFrame({
            "Feature": [clf.feature_names_in_[i] for i in top_idx],
            "Importance": feature_importance[top_idx]
        })

        st.write("### 🔍 Top 10 Contributing Features")
        st.dataframe(top_features)

        # Emotion contribution
        st.write("### 😀 Emotion Pattern Summary")
        st.write("Higher volatility or unusual emotion spikes may indicate deception.")

        # Body-language explanation
        st.write("### 🧍 Body-Language Summary")
        st.write(
            "- High movement variance → fidgeting\n"
            "- Shoulder instability → nervousness\n"
            "- Low head stability → distraction\n"
            "- Hand-to-face proximity → classic deception cue"
        )

    except Exception as e:
        st.error(f"Processing failed: {e}")
        st.stop()
