import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import torch
import cv2
import numpy as np

# PyTorch 2.6+ compat: Monkey-patch torch.load to default weights_only=False
# Required because ultralytics calls torch.load without arguments.
_original_load = torch.load
def safe_load_shim(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load_shim

# Module-level cache (works for both Streamlit AND FastAPI)
_pose_model = None

def get_pose_model():
    """Lazily load and cache the YOLO pose model once for the lifetime of the app."""
    global _pose_model
    if _pose_model is not None:
        return _pose_model
    from ultralytics import YOLO
    _pose_model = YOLO("yolov8n-pose.pt")
    return _pose_model


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

    kp_obj = results[0].keypoints

    if kp_obj is None:
        return np.zeros(51, dtype=np.float32)

    if kp_obj.data is None or kp_obj.data.shape[0] == 0:
        return np.zeros(51, dtype=np.float32)

    kpts = kp_obj.data[0].cpu().numpy()  # shape: (17, 3)
    return kpts.reshape(-1).astype(np.float32)  # flatten to (51,)
