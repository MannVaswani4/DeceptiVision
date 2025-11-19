# DeceptiVision

**DeceptiVision** is a multimodal deception-detection research system designed to classify **truthful** vs **deceptive** behavior using **facial micro-expressions, body language, and video-based behavioral cues**.  
The repository provides a complete, reproducible pipeline including **video preprocessing, frame extraction, face detection, pose estimation, dataset preparation, model training, and inference**.

* * *

## Why DeceptiVision Is Useful

-   Provides a **fully defined, end-to-end workflow** for building deception-detection models from raw videos.
    
-   Includes **deterministic preprocessing scripts** for converting raw interview videos into structured datasets.
    
-   Offers **baseline deep-learning models** for face-emotion recognition and pose-based body language analysis.
    
-   Enforces a **consistent dataset format**, allowing researchers to extend the system with new videos or modalities (e.g., audio, biosignals).
    
-   Designed for **reproducible research**, making it compatible with academic, forensic, and behavioral-analysis use cases.
    

* * *

## Dataset

DeceptiVision uses **three structured dataset components**, all located in the `data/` directory.

### 1\. deception\_dataset.csv

Path: `data/deception_dataset.csv`  
This CSV file is the **authoritative index** for the project.  
It contains one row per trial with the following columns:

| Column | Description |
| --- | --- |
| `trial_id` | Unique trial identifier |
| `frames_path` | Path to the folder containing extracted frames for the trial |
| `label` | `truth` or `lie` |
| `participant_id` | (Optional) Participant identifier |
| `notes` | (Optional) Metadata or annotation notes |

Every training and inference script in the repository relies on this manifest.

* * *

### 2\. Video Dataset

Paths:

-   `videos/truth/`
    
-   `videos/lie/`
    

These folders store all raw interview videos.  
Each file corresponds to exactly one deception-classification trial.  
The naming convention is up to the user, but each video **must** be referenced in `deception_dataset.csv` after frame extraction.

* * *

### 3\. Frame Dataset

Path: `data/frames/`

Each trial has a **dedicated folder** inside `data/frames/`, containing sequential frames extracted from the corresponding video.

Example structure:

`data/frames/     trial_truth_001/         frame_0001.jpg         frame_0002.jpg         ...     trial_lie_001/         frame_0001.jpg         frame_0002.jpg         ...`

Every folder listed in `deception_dataset.csv` must exist here.

* * *

### 4\. FER2013 Dataset (for emotion pretraining)

Path: `data/raw/fer2013/`

FER2013 is used to **pretrain the facial emotion recognition backbone**.  
Place the original dataset files inside the folder above.  
The training script automatically loads it when emotion-model pretraining is enabled.

* * *

## Preprocessing Pipeline

DeceptiVision includes **deterministic preprocessing scripts** to ensure all users obtain identical outputs.

### 1\. Frame Extraction

Script: `src/video_to_frames.py`  
Function: Converts each raw video into sequential frames at the specified frame rate.

Command:

`python src/video_to_frames.py --input <VIDEO_PATH> --output <FRAME_OUTPUT_DIR>`

* * *

### 2\. Face Detection & Emotion Feature Extraction

Script: `src/feature_extractor.py`  
Function:

-   Detects faces in every frame
    
-   Normalizes, aligns, and crops face regions
    
-   Extracts emotion embeddings using the model in `models/emotion_cnn.pth`
    

Command:

`python src/feature_extractor.py --frames-dir data/frames/ --out-dir data/processed/faces/ --model models/emotion_cnn.pth`

* * *

### 3\. Body Language / Pose Feature Extraction

Script: `src/body_language_yolo.py`  
Function:

-   Uses YOLO-Pose (`notebooks/yolov8n-pose.pt`) to compute body-keypoint features
    
-   Outputs per-frame pose vectors to `data/processed/pose/`
    

Command:

`python src/body_language_yolo.py --frames-dir data/frames/ --out-dir data/processed/pose/ --yolo-model notebooks/yolov8n-pose.pt`

* * *

## Training Pipeline

### Facial Emotion Fine-Tuning

Script: `training/train_face_finetune.py`  
Function:

-   Loads FER2013 (optional pretraining)
    
-   Loads deception frames listed in `deception_dataset.csv`
    
-   Fine-tunes the emotion recognition backbone for lie/truth classification
    

Command:

`python training/train_face_finetune.py \   --manifest data/deception_dataset.csv \   --frames-root data/frames \   --output-dir models/face_finetune \   --epochs 20 \   --batch-size 32`

The resulting model is saved inside `models/face_finetune/`.

* * *

## Inference / Demo

### Command-Line Inference

Script: `app.py`  
Function: Runs deception detection on a new video using the trained model.

Command:

`python app.py --video <VIDEO_PATH> --model models/emotion_cnn.pth`

Output includes:

-   predicted label (`truth` or `lie`)
    
-   confidence score
    
-   optional JSON output in `outputs/`
    

* * *

## Environment Setup (Deterministic)

### 1\. Python Version

DeceptiVision requires **Python 3.8–3.11**.

### 2\. Virtual Environment Setup

`python3 -m venv .venv source .venv/bin/activate python -m pip install --upgrade pip pip install -r requirements.txt`

* * *

## Project Structure

`. ├── app.py                     # Inference / demo entry point ├── data/                      # Datasets (raw, frames, processed, manifests) │   ├── deception_dataset.csv │   ├── frames/ │   ├── processed/ │   └── raw/ ├── models/                    # Pretrained and fine-tuned model weights ├── notebooks/                 # Research and development notebooks ├── src/                       # Core data processing and feature extraction │   ├── video_to_frames.py │   ├── feature_extractor.py │   └── body_language_yolo.py ├── training/                  # Training scripts │   └── train_face_finetune.py └── requirements.txt`

* * *

## Usage Examples

### Python API Example

`from src.feature_extractor import predict_trial  result = predict_trial(     frames_dir="data/frames/trial_truth_001",     model_path="models/emotion_cnn.pth" )  print(result["label"], result["confidence"])`

### Combined CLI Workflow

`python src/video_to_frames.py --input videos/truth/new.mp4 --output data/frames/new_trial python src/feature_extractor.py --frames-dir data/frames/new_trial --out-dir data/processed/new_trial --model models/emotion_cnn.pth python app.py --frames data/processed/new_trial --model models/emotion_cnn.pth`