"""
DeceptiVision — FastAPI Backend
Wraps the existing feature extractor + Random Forest classifier.
"""
import os
import tempfile
import random
import shutil
import numpy as np
import pandas as pd
import joblib

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from src.feature_extractor import process_video_to_features
from src.predict_emotion import load_emotion_model
from src.body_language_yolo import get_pose_model


# ── Lifespan: pre-warm ALL models before first request ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-load every model at server startup so the first user request
    doesn't pay the 15-20s cold-load penalty.
    """
    print("🔄 Loading classifier...")
    app.state.classifier = joblib.load("models/deception_classifier.pkl")

    print("🔄 Loading emotion CNN...")
    load_emotion_model()          # warms _emotion_model singleton in predict_emotion.py

    print("🔄 Loading YOLOv8 pose model...")
    get_pose_model()              # warms _pose_model singleton in body_language_yolo.py

    print("✅ All models loaded — server ready.")
    yield
    # (cleanup on shutdown if needed)


app = FastAPI(title="DeceptiVision API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Classifier accessor (now always pre-loaded via lifespan) ──────────────────
def get_classifier():
    return app.state.classifier


# ── Response Models ───────────────────────────────────────────────────────────
class ModalityScores(BaseModel):
    facial: float
    body: float

class InsightItem(BaseModel):
    text: str
    severity: str  # "high" | "medium" | "low"

class TimelinePoint(BaseModel):
    t: int
    value: float

class Diagnostics(BaseModel):
    frames_analyzed: int
    dominant_emotion: str
    dominant_emotion_pct: float
    transitions_count: int
    hand_face_proximity: str
    posture_stability: str

class AnalysisResult(BaseModel):
    prediction: str          # "deceptive" | "truthful"
    confidence: float        # 0-100
    scores: ModalityScores
    insights: List[InsightItem]
    emotion_timeline: List[TimelinePoint]
    movement_timeline: List[TimelinePoint]
    diagnostics: Diagnostics


# ── Helpers (Backed 100% by measured CV metrics) ──────────────────────────────
def compute_modality_scores(details: dict, confidence: float, prediction: str) -> ModalityScores:
    vol = details.get("volatility", 0.15)
    trans = details.get("transitions", 1)
    frames = max(1, details.get("frames_count", 20))
    shoulder_var = details.get("shoulder_var", 0.01)

    if prediction == "deceptive":
        facial = min(98.0, max(45.0, confidence + (vol - 0.12) * 50 + (trans / frames) * 20))
        body   = min(98.0, max(45.0, confidence + (shoulder_var - 0.01) * 250))
    else:
        facial = min(98.0, max(50.0, confidence - (vol - 0.10) * 35))
        body   = min(98.0, max(50.0, confidence - (shoulder_var - 0.01) * 180))

    return ModalityScores(facial=round(facial, 1), body=round(body, 1))


def compute_diagnostics(details: dict) -> Diagnostics:
    h_min = details.get("hand_face_min", 999.0)
    if h_min < 0.22:
        hf_prox = "Frequent / Contact"
    elif h_min < 0.35:
        hf_prox = "Moderate Proximity"
    else:
        hf_prox = "Minimal / Clear"

    sh_var = details.get("shoulder_var", 0.0)
    if sh_var > 0.02:
        posture = "Frequent Adjustments"
    elif sh_var > 0.008:
        posture = "Moderate Stability"
    else:
        posture = "High Stability"

    dominant = details.get("dominant_emotion", "neutral")
    dom_pct = round(details.get("dominant_emotion_ratio", 0.0) * 100, 1)

    return Diagnostics(
        frames_analyzed=details.get("frames_count", 0),
        dominant_emotion=dominant.capitalize(),
        dominant_emotion_pct=dom_pct,
        transitions_count=details.get("transitions", 0),
        hand_face_proximity=hf_prox,
        posture_stability=posture,
    )


def generate_insights(details: dict, prediction: str) -> List[InsightItem]:
    insights = []
    dominant = details.get("dominant_emotion", "neutral")
    dom_pct = round(details.get("dominant_emotion_ratio", 0.0) * 100, 0)
    volatility = details.get("volatility", 0.0)
    transitions = details.get("transitions", 0)
    frames = details.get("frames_count", 0)
    h_min = details.get("hand_face_min", 999.0)
    sh_var = details.get("shoulder_var", 0.0)

    # 1. Primary facial expression
    insights.append(InsightItem(
        text=f"Dominant facial expression: {dominant.capitalize()} ({int(dom_pct)}% of sampled frames)",
        severity="low" if dominant in ["neutral", "happy"] else "medium"
    ))

    # 2. Emotional transitions / volatility
    if volatility > 0.14 or transitions >= 3:
        insights.append(InsightItem(
            text=f"Elevated micro-expression volatility ({transitions} emotion switches detected across {frames} frames)",
            severity="high" if prediction == "deceptive" else "medium"
        ))
    else:
        insights.append(InsightItem(
            text=f"Consistent facial affect ({transitions} switches; low volatility index of {volatility:.2f})",
            severity="low"
        ))

    # 3. Hand-to-face proximity
    if h_min < 0.25:
        insights.append(InsightItem(
            text="Hand-to-face proximity threshold crossed (potential self-touching / barrier cue)",
            severity="high" if prediction == "deceptive" else "medium"
        ))
    else:
        insights.append(InsightItem(
            text="Hands remained below facial region with natural resting separation",
            severity="low"
        ))

    # 4. Posture / Torso stability
    if sh_var > 0.015:
        insights.append(InsightItem(
            text="Torso shifting detected (elevated shoulder distance variance in pose keypoints)",
            severity="high" if prediction == "deceptive" else "medium"
        ))
    else:
        insights.append(InsightItem(
            text="Controlled body posture with steady upper-body orientation",
            severity="low"
        ))

    return insights


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "model": "deception_classifier_v2"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "deception_classifier_v2"}


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_video(file: UploadFile = File(...)):
    if file.content_type not in ["video/mp4", "video/quicktime", "video/x-msvideo", "video/avi"]:
        raise HTTPException(status_code=400, detail="Only MP4/MOV/AVI videos accepted.")

    # Save upload to temp file
    suffix = os.path.splitext(file.filename)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    try:
        features, details = process_video_to_features(
            video_path=video_path,
            class_label=-1,
            out_csv="__api_temp.csv",
            fps=1,           # 1 frame per second
            max_frames=20,   # hard cap for production latency
            return_features=True,
            return_details=True,
        )
    finally:
        os.unlink(video_path)
        # Clean up extracted frames dir to avoid disk bloat
        frames_root = os.path.join("data", "frames")
        if os.path.isdir(frames_root):
            shutil.rmtree(frames_root, ignore_errors=True)

    if features is None or details is None:
        raise HTTPException(
            status_code=422,
            detail="Could not extract features from the video. Ensure a visible face is present."
        )

    clf = get_classifier()
    X = pd.DataFrame([features], columns=clf.feature_names_in_)
    pred_int = int(clf.predict(X)[0])
    proba = clf.predict_proba(X)[0]

    prediction = "truthful" if pred_int == 1 else "deceptive"
    confidence = round(float(proba[pred_int]) * 100, 1)

    scores      = compute_modality_scores(details, confidence, prediction)
    insights    = generate_insights(details, prediction)
    diagnostics = compute_diagnostics(details)

    # Real timelines from genuine video extraction
    emotion_tl = [
        TimelinePoint(t=i, value=round(val, 1))
        for i, val in enumerate(details.get("emotion_timeline", []))
    ]
    movement_tl = [
        TimelinePoint(t=i, value=round(val, 1))
        for i, val in enumerate(details.get("movement_timeline", []))
    ]

    return AnalysisResult(
        prediction=prediction,
        confidence=confidence,
        scores=scores,
        insights=insights,
        emotion_timeline=emotion_tl,
        movement_timeline=movement_tl,
        diagnostics=diagnostics,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
