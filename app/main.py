
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict_ecg

app = FastAPI(
    title="ECG Arrhythmia Classification API",
    description="ECG heartbeat classification using PyTorch LSTM.",
    version="1.0.0"
)


class ECGRequest(BaseModel):
    signal: list[float]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "LSTM Plain"
    }


@app.post("/predict")
def predict(request: ECGRequest):
    try:
        return predict_ecg(request.signal)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


import gradio as gr
from app.frontend import demo

app = gr.mount_gradio_app(
    app,
    demo,
    path="/"
)
