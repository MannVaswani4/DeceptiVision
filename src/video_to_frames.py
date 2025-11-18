import cv2
import os

def extract_frames(video_path, output_folder, fps=2):
    """
    Extract frames from video at given FPS.
    Saves frames as .jpg inside output_folder.
    """

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30  # fallback

    frame_interval = int(video_fps / fps)

    frame_id = 0
    saved_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_id}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_id += 1

        frame_id += 1

    cap.release()
    print(f"✅ Extracted {saved_id} frames from {video_path} → {output_folder}")
