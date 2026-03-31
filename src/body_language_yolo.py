import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import torch
import cv2
import numpy as np
import streamlit as st

# PyTorch 2.6+ compat: Monkey-patch torch.load to default weights_only=False
# This is required because ultralytics calls torch.load without arguments,
# which now defaults to safe-only mode, breaking loading of complex models.
_original_load = torch.load
def safe_load_shim(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load_shim


@st.cache_resource(show_spinner=False)
def get_pose_model():
    """Lazily load and cache the YOLO pose model once for the lifetime of the app."""
    from ultralytics import YOLO
    return YOLO("yolov8n-pose.pt")


def extract_yolo_pose(image_path):
    """
    Returns 51 pose values (17 keypoints × [x, y, confidence]).
    If no person detected → returns zeros.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not found:", image_path)
        return np.zeros(51, dtype=np.float32)

    model = get_pose_model()
    results = model(img, verbose=False)

    kp_obj = results[0].keypoints  # Keypoints object

    # Case 1: YOLO produced no keypoints at all
    if kp_obj is None:
        return np.zeros(51, dtype=np.float32)

    # Case 2: YOLO produced keypoints object but no persons
    if kp_obj.data is None or kp_obj.data.shape[0] == 0:
        return np.zeros(51, dtype=np.float32)

    # Extract tensor for first detected person
    kpts = kp_obj.data[0].cpu().numpy()  # shape: (17, 3)

    return kpts.reshape(-1).astype(np.float32)  # flatten to (51,)
