
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.predict import predict_ecg as predict_signal


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_PATH = PROJECT_ROOT / "app" / "examples.csv"

CLASS_NAMES = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown"
}

CLASS_IDS = {name: idx for idx, name in CLASS_NAMES.items()}

# Small, balanced sample dataset.
examples_df = pd.read_csv(EXAMPLES_PATH, header=None)


def make_waveform(signal):
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(signal, linewidth=1.4)
    ax.set_title("ECG Heartbeat Waveform")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    return fig


def make_probability_chart(probabilities):
    return pd.DataFrame({
        "Class": list(probabilities.keys()),
        "Probability (%)": list(probabilities.values())
    })


def predict_sample(class_id):

    class_id = int(class_id)

    matching_samples = examples_df[
        examples_df[187] == class_id
    ]

    if matching_samples.empty:
        raise gr.Error("No examples available for this class.")

    # Randomly select one example of the requested class.
    row = matching_samples.sample(n=1).iloc[0]

    signal = row.iloc[:187].to_numpy(dtype=np.float32)
    actual_label = CLASS_NAMES[class_id]

    result = predict_signal(signal)

    predicted_class = result["predicted_class"]
    predicted_probability = result["confidence"]

    is_correct = predicted_class == actual_label

    correctness = (
        "Correct prediction"
        if is_correct
        else "Incorrect prediction"
    )

    prediction_markdown = f"""
## Prediction Result

# {predicted_class}

**Predicted probability:** {predicted_probability:.2f}%

**Actual dataset label:** {actual_label}

**Evaluation:** {"Correct" if is_correct else "Incorrect"}
"""

    waveform = make_waveform(signal)

    probability_chart = make_probability_chart(
        result["probabilities"]
    )

    return waveform, prediction_markdown, probability_chart


def read_uploaded_csv(file_path):

    if file_path is None:
        raise gr.Error("Please upload a CSV file.")

    try:
        df = pd.read_csv(file_path, header=None)

        # Support CSV files with a header row.

        not df.apply(
            lambda column: pd.to_numeric(
                column, errors="coerce"
            ).notna().all()
        ).all()

        df = pd.read_csv(file_path, header=0)

        if len(df) == 0:
            raise gr.Error("The uploaded CSV is empty.")

        if len(df) > 100:
            raise gr.Error(
                "Maximum upload limit is 100 ECG heartbeats per CSV."
            )

        if df.shape[1] != 187:
            raise gr.Error(
                f"Each heartbeat must contain exactly 187 values. "
                f"Your CSV contains {df.shape[1]} columns."
            )

        df = df.apply(pd.to_numeric, errors="raise")

        signals = df.to_numpy(dtype=np.float32)

        if not np.isfinite(signals).all():
            raise gr.Error(
                "CSV contains missing or infinite values."
            )

        return signals

    except gr.Error:
        raise

    except Exception as error:
        raise gr.Error(
            f"Invalid CSV file. Please check its format. Details: {error}"
        )


def predict_uploaded_csv(file_path):

    signals = read_uploaded_csv(file_path)

    results = []

    for index, signal in enumerate(signals):

        result = predict_signal(signal)

        results.append({
            "Heartbeat": index + 1,
            "Predicted Class": result["predicted_class"],
            "Predicted Probability (%)": result["confidence"]
        })

    results_df = pd.DataFrame(results)

    # Display the first uploaded heartbeat's waveform.
    first_waveform = make_waveform(signals[0])

    status = (
        f"Successfully analyzed {len(signals)} heartbeat(s)."
    )

    return results_df, first_waveform, status


custom_css = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
}

#subtitle {
    text-align: center;
    opacity: 0.75;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css,
    title="ECG Arrhythmia Classifier"
) as demo:

    gr.Markdown(
        "# ECG Arrhythmia Classification",
        elem_id="main-title"
    )

    gr.Markdown(
        "PyTorch LSTM-based five-class heartbeat classification",
        elem_id="subtitle"
    )

    gr.Markdown(
        """
        **Educational project, not a medical device.**
        Predictions must not be used for medical diagnosis.
        """
    )

    gr.Markdown("---")

    
    with gr.Tab("Try Sample ECGs"):

        gr.Markdown(
            "Select a heartbeat category to randomly sample "
            "a labelled example from the test dataset."
        )

        sample_buttons = []

        with gr.Row():
            for class_id, class_name in CLASS_NAMES.items():
                button = gr.Button(
                    f"Random {class_name}",
                    variant="secondary"
                )
                sample_buttons.append((button, class_id))

        sample_waveform = gr.Plot(label="ECG Waveform")

        sample_prediction = gr.Markdown(
            "Choose a sample class to begin."
        )

        sample_probabilities = gr.BarPlot(
            x="Class",
            y="Probability (%)",
            title="Predicted Class Probabilities",
            y_lim=[0, 100]
        )

        for button, class_id in sample_buttons:
            button.click(
                fn=lambda cid=class_id: predict_sample(cid),
                inputs=[],
                outputs=[
                    sample_waveform,
                    sample_prediction,
                    sample_probabilities
                ]
            )
            
            
    with gr.Tab("Upload ECG CSV"):

        gr.Markdown(
            """
            Upload a CSV containing one or more ECG heartbeats.

            - Exactly 187 numeric values per row.
            - Maximum 100 rows.
            - No actual class-label column.
            - Signals must use the same preprocessing scale as training.
            """
        )

        uploaded_file = gr.File(
            label="Upload ECG CSV",
            file_types=[".csv"],
            type="filepath"
        )

        upload_button = gr.Button(
            "Analyze Uploaded ECGs",
            variant="primary"
        )

        upload_status = gr.Markdown()

        upload_results = gr.Dataframe(
            label="Prediction Results",
            interactive=False
        )

        upload_waveform = gr.Plot(
            label="First Uploaded Heartbeat"
        )

        upload_button.click(
            fn=predict_uploaded_csv,
            inputs=uploaded_file,
            outputs=[
                upload_results,
                upload_waveform,
                upload_status
            ]
        )

    gr.Markdown("---")

    gr.Markdown(
        """
        **Disclaimer:** This is an educational machine learning
        demonstration, not a medical device.
        """
    )
