from src.body_language_yolo import extract_yolo_pose     # <-- NEW
from pathlib import Path
import numpy as np
import csv
import os

def process_video_to_features(video_path, class_label, out_csv, fps=2):
    """
    Extracts:
        - Emotion timeline
        - Body-language timeline (YOLO Pose)
    Then creates:
        - 23 emotion features
        - 7 body-language features
    Then writes one row to CSV.
    """

    video_name = Path(video_path).stem
    frame_dir = Path(f"../data/frames/{video_name}")
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames from video
    extract_frames(str(video_path), str(frame_dir), fps=fps)

    # List frames
    frames = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith((".jpg", ".png"))])

    if len(frames) == 0:
        print(f"⚠️ No frames extracted for {video_name}")
        return False

    emotion_list = []
    pose_list = []

    # ------------ PER FRAME FEATURE EXTRACTION ------------
    for f in frames:
        frame_path = str(frame_dir / f)

        # Emotion (your existing model)
        emot = predict_emotion(frame_path)
        emotion_list.append(emot)

        # Pose keypoints (YOLOv8)
        pose = extract_yolo_pose(frame_path)
        pose_list.append(pose)

    # Convert to numpy arrays
    emotion_timeline = np.array(emotion_list)   # (N_frames, 7)
    pose_timeline = np.array(pose_list)         # (N_frames, 51)

    # ------------ EMOTION FEATURES (YOUR EXISTING LOGIC) ------------
    dominant = np.argmax(emotion_timeline, axis=1)
    unique, counts = np.unique(dominant, return_counts=True)
    ratios = {e: 0 for e in emotion_labels}
    for idx, c in zip(unique, counts):
        ratios[emotion_labels[idx]] = c / len(dominant)

    transitions = int(np.sum(dominant[:-1] != dominant[1:]))
    volatility = float(np.mean(np.abs(np.diff(emotion_timeline, axis=0))))

    peak_vals = emotion_timeline.max(axis=0)
    var_vals = emotion_timeline.var(axis=0)

    emotion_features = []
    for e in emotion_labels:
        emotion_features.append(float(ratios[e]))

    emotion_features += [float(transitions), float(volatility)]
    emotion_features += [float(v) for v in peak_vals]
    emotion_features += [float(v) for v in var_vals]

    # ------------ BODY-LANGUAGE FEATURES (NEW) ------------
    # movement intensity: frame-to-frame change of all keypoints
    movement = np.linalg.norm(np.diff(pose_timeline, axis=0), axis=1)
    movement_mean = float(movement.mean())
    movement_var = float(movement.var())
    movement_max = float(movement.max())

    # shoulder width stability (keypoints 5 & 6)
    left_shoulder  = pose_timeline[:, 5*3:5*3+2]
    right_shoulder = pose_timeline[:, 6*3:6*3+2]
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
    shoulder_var = float(shoulder_dist.var())

    # head movement (nose index = 0)
    nose = pose_timeline[:, 0:3]
    head_speed = np.linalg.norm(np.diff(nose, axis=0), axis=1)
    head_var = float(head_speed.var())

    # hand-to-face (keypoints 9=left wrist, 10=right wrist)
    left_wrist = pose_timeline[:, 9*3:9*3+3]
    right_wrist = pose_timeline[:, 10*3:10*3+3]
    nose_3d = pose_timeline[:, 0:3]

    lw_dist = np.linalg.norm(left_wrist - nose_3d, axis=1)
    rw_dist = np.linalg.norm(right_wrist - nose_3d, axis=1)

    hand_face_min = float(min(lw_dist.min(), rw_dist.min()))
    hand_face_mean = float((lw_dist.mean() + rw_dist.mean()) / 2)

    body_features = [
        movement_mean, movement_var, movement_max,
        shoulder_var, head_var,
        hand_face_min, hand_face_mean
    ]

    # ------------ COMBINE ALL FEATURES ------------
    features = emotion_features + body_features

    # Write CSV header if necessary
    header = [f"f{i}" for i in range(len(features))] + ["label"]
    file_exists = os.path.exists(out_csv)

    if not file_exists:
        with open(out_csv, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # Write new row
    row = features + [int(class_label)]
    with open(out_csv, "a", newline="") as f:
        csv.writer(f).writerow(row)

    print(f"✅ Processed {video_name} | Features: {len(features)} | Label: {class_label}")
    return features 
