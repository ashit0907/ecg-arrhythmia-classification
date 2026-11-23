
import numpy as np
import torch


def preprocess_signal(signal):
    """
    Validate and convert a single ECG signal
    into a PyTorch tensor.

    Expected input: 187 ECG values
    Output shape: (1, 187)
    """

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 1:
        raise ValueError("ECG signal must be one-dimensional.")

    if len(signal) != 187:
        raise ValueError(
            f"Expected 187 values, received {len(signal)}."
        )

    if not np.isfinite(signal).all():
        raise ValueError("ECG signal contains invalid values.")

    signal_tensor = torch.tensor(
        signal,
        dtype=torch.float32
    ).unsqueeze(0)

    return signal_tensor
