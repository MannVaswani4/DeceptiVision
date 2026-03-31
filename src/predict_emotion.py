import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import streamlit as st

from src.models.face_cnn import FaceCNN

# Emotion labels (match your original dataset)
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Device — MPS is macOS-only; fall back to CPU on Linux / Streamlit Cloud
def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS only available on Apple Silicon Macs
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = _get_device()

# Transform (same as training)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])


@st.cache_resource(show_spinner=False)
def load_emotion_model(model_path="models/emotion_cnn.pth"):
    """Load the FaceCNN model once and cache it for all sessions."""
    model = FaceCNN(num_emotions=7).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_emotion(image_path):
    """Returns a probability vector (7 emotions)."""
    model = load_emotion_model()

    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]

    return probs
