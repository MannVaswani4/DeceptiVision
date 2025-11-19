import streamlit as st
import tempfile
import os
import numpy as np
import pandas as pd
import joblib

from src.feature_extractor import process_video_to_features

st.title("DeceptiVision — Lie Detection AI")
st.write("Upload a short video. The system analyzes:")
st.write("• Facial emotion micro-expressions")
st.write("• Body language (YOLO pose)")
st.write("• Motion patterns")
st.write("Then predicts Lie / Truth.")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    st.video(uploaded_video)

    # Save temp video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(uploaded_video.read())
        video_path = temp.name

    st.write("Processing video...")

    # Extract features
    features = process_video_to_features(
        video_path=video_path,
        class_label=-1,               # unknown
        out_csv="__streamlit_temp.csv",
        fps=2,
        return_features=True          # YOU MUST ADD THIS IN feature_extractor
    )

    if features is None:
        st.error("Failed to extract features.")
    else:
        st.success("Features extracted!")

        # Load classifier
        clf = joblib.load("models/deception_classifier.pkl")

        X = pd.DataFrame([features], columns=clf.feature_names_in_)
        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]

        label = "TRUTH" if pred == 1 else "LIE"

        st.header(f"Prediction: **{label}**")

        st.write("Confidence:")
        st.progress(float(max(proba)))

        # Detailed explanation
        st.subheader("Why this decision?")
        st.write(f"Truth probability: {proba[1]:.3f}")
        st.write(f"Lie probability: {proba[0]:.3f}")
        st.write("The model used:")
        st.write("• Emotion instability")
        st.write("• Facial micro-expressions")
        st.write("• Body movement patterns")
        st.write("• Head movement variance")
        st.write("• Hand-to-face behaviors")
