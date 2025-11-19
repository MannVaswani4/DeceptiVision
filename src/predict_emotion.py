import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os

from src.models.face_cnn import FaceCNN

# Emotion labels (match your original dataset)
emotion_labels = ["angry","disgust","fear","happy","sad","surprise","neutral"]

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Transform (same as training)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Load pretrained model once
_model = None

def load_emotion_model(model_path="../models/emotion_cnn.pth"):
    global _model
    if _model is None:
        model = FaceCNN(num_emotions=7).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        _model = model
    return _model


def predict_emotion(image_path):
    """Returns a probability vector (7 emotions)."""

    model = load_emotion_model()

    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]

    return probs
