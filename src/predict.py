
from pathlib import Path

import torch

from src.model import ECG_RNN
from src.preprocessing import preprocess_signal


# Class labels
CLASS_NAMES = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown"
}


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_plain.pt"


# Device configuration
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = ECG_RNN(
        cell="lstm",
        hidden=64,
        layers=2,
        dropout=0.2,
        n_classes=5
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# Load model once when this module is imported
model = load_model()


def predict_ecg(signal):

    signal_tensor = preprocess_signal(signal)
    signal_tensor = signal_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(signal_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()

    probability_dict = {
        CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": CLASS_NAMES[predicted_class],
        "confidence": round(
            probabilities[predicted_class].item() * 100, 2
        ),
        "probabilities": probability_dict
    }
