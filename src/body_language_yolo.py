from ultralytics import YOLO
import cv2
import numpy as np

pose_model = YOLO("yolov8n-pose.pt")

def extract_yolo_pose(image_path):
    """
    Returns 51 pose values (17 keypoints × [x, y, confidence]).
    If no person detected → returns zeros.
    """

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not found:", image_path)
        return np.zeros(51, dtype=np.float32)

    results = pose_model(img, verbose=False)

    kp_obj = results[0].keypoints  # Keypoints object

    # Case 1: YOLO produced no keypoints at all
    if kp_obj is None:
        return np.zeros(51, dtype=np.float32)

    # Case 2: YOLO produced keypoints object but no persons
    if kp_obj.data is None or kp_obj.data.shape[0] == 0:
        return np.zeros(51, dtype=np.float32)

    # Extract tensor for first detected person
    kpts = kp_obj.data[0].cpu().numpy()  # shape: (17,3)

    return kpts.reshape(-1).astype(np.float32)  # flatten to (51,)
