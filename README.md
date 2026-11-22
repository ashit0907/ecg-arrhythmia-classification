# ECG Heartbeat Arrhythmia Classification: 1D CNN vs LSTM vs GRU

A PyTorch-based ECG heartbeat classification project comparing a 1D CNN, Bidirectional LSTM, and Bidirectional GRU against traditional machine learning baselines. The project investigates class imbalance, model architecture choices, evaluation metrics, and generalization limitations.

> **Educational project — not a medical device.** This model is not intended for diagnosis, clinical decision-making, or patient monitoring.

**Demo:** Coming soon — Hugging Face Spaces

---

## Overview

This project classifies individual ECG heartbeat segments into five categories using traditional machine learning and deep learning approaches.

The primary objective is to compare different architectures on an imbalanced classification problem and investigate whether deep learning provides meaningful improvements over traditional machine learning.

### Key results

| Metric | Result |
|---|---:|
| Best validation Macro-F1 | 94.43% |
| Best validation accuracy | 99.02% |
| Selected model | Bidirectional LSTM (Plain) |
| Final test accuracy | 98.70% |
| Final test Macro-F1 | 92.89% |
| Final test balanced accuracy | 91.30% |
| Best traditional ML Macro-F1 | 90.37% (Random Forest) |

The Bidirectional LSTM trained with standard cross-entropy was selected based on validation Macro-F1 and evaluated once on the separate test dataset.

**Main findings:**

- Accuracy alone is misleading on highly imbalanced ECG data.
- Random Forest provides a strong traditional ML baseline.
- Plain LSTM and GRU architectures achieved the highest validation Macro-F1.
- Class-weighted loss improved minority-class recall in some experiments but reduced overall Macro-F1.
- The publicly available dataset split may contain patient-level leakage, limiting the reliability of generalization claims.

---

## Dataset

**Dataset:** [ECG Heartbeat Categorization Dataset — Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

The dataset is derived from the MIT-BIH Arrhythmia Database and contains heartbeat segments represented as fixed-length numerical signals.

| Class | Label | Description |
|---|---|---|
| 0 | N | Normal heartbeat |
| 1 | S | Supraventricular ectopic beat |
| 2 | V | Ventricular ectopic beat |
| 3 | F | Fusion of ventricular and normal beat |
| 4 | Q | Unknown or paced beat |

Each sample contains 187 numerical signal values followed by the class label. The signals are scaled and shorter sequences are zero-padded.

### Dataset files

- `mitbih_train.csv`: approximately 87,554 heartbeat samples
- `mitbih_test.csv`: approximately 21,892 heartbeat samples

The training dataset was divided into training and validation subsets using a stratified 85/15 split with a fixed random seed.

| Subset | Samples |
|---|---:|
| Training | 74,420 |
| Validation | 13,133 |
| Total used | 87,553 |

The saved split indices are stored under `data/processed/` to maintain consistency across experiments.

The Kaggle test dataset was kept separate from model training and validation and was used for final evaluation of the selected model.

### Class imbalance

The dataset is heavily dominated by Normal heartbeat samples. The minority classes, especially Fusion and Supraventricular beats, contain substantially fewer examples.

This imbalance makes accuracy an insufficient evaluation metric. Macro-F1 and balanced accuracy are used to assess performance across all classes.

---

## Important limitation: potential patient leakage

The public dataset files do not provide patient identifiers for the individual heartbeat samples.

The commonly used train/test division is performed at the heartbeat level rather than through a verified patient-wise split. Consequently, beats originating from the same patient may potentially appear in both training and test datasets.

This means the model may encounter heartbeat patterns from patients it has indirectly seen during training.

A patient-wise or inter-patient evaluation would provide a more realistic assessment of generalization to unseen individuals.

**The results reported here should therefore not be interpreted as evidence of patient-independent performance or clinical reliability.**

---

## Methodology

The project follows a consistent experimental workflow:

1. Exploratory data analysis and signal visualization.
2. Stratified training/validation split.
3. Traditional machine learning baselines.
4. Deep learning model development using PyTorch.
5. Plain versus class-weighted loss experiments.
6. Model selection using validation Macro-F1.
7. Final evaluation on the separate test dataset.

### Traditional machine learning baselines

Three traditional machine learning models were evaluated using the 187 ECG signal values as input features.

| Model | Approach | Hyperparameter tuning |
|---|---|---|
| Dummy Classifier | Majority-class prediction | None |
| Logistic Regression | Linear classification | RandomizedSearchCV |
| Random Forest | Ensemble of decision trees | RandomizedSearchCV |

The Dummy Classifier provides a reference performance by predicting the majority class.

Logistic Regression and Random Forest were optimized using `RandomizedSearchCV`, with hyperparameter search conducted on the training portion of the dataset. The validation set was reserved for evaluating the resulting models.

This provides a stronger baseline comparison by evaluating tuned traditional machine learning models against the deep learning architectures.

### Deep learning architectures

#### 1D CNN

The CNN extracts local temporal features from ECG signals using convolutional layers.

Architecture:

- Conv1D: 1 → 32 channels, kernel size 7
- Batch Normalization
- ReLU
- Max Pooling
- Conv1D: 32 → 64 channels, kernel size 5
- Batch Normalization
- ReLU
- Max Pooling
- Conv1D: 64 → 128 channels, kernel size 3
- Batch Normalization
- ReLU
- Max Pooling
- Adaptive Average Pooling
- Linear classification layer

**Total trainable parameters: 36,317**

#### Bidirectional LSTM

The LSTM processes the heartbeat as a sequence and captures temporal relationships in both forward and backward directions.

Architecture:

- Input size: 1
- Hidden size: 64
- Number of layers: 2
- Bidirectional: Yes
- Dropout: 0.2
- Mean pooling and max pooling across time
- Linear classification head

The mean and max pooled representations are concatenated before classification.

**Total trainable parameters: 134,917**

#### Bidirectional GRU

The GRU uses the same overall design and configuration as the LSTM, replacing the LSTM recurrent cells with GRU cells.

**Total trainable parameters: 101,509**

### Training configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Maximum epochs | 50 |
| Early stopping | Patience 8 |
| Gradient clipping | 1.0 |
| Learning rate scheduler | ReduceLROnPlateau |
| Model selection metric | Validation Macro-F1 |
| Random seed | 42 |

The best checkpoint for each experiment was saved based on validation Macro-F1 rather than the final training epoch.

---

## Class imbalance experiments

Each deep learning architecture was trained twice:

1. **Plain:** Standard CrossEntropyLoss.
2. **Weighted:** CrossEntropyLoss with class weights calculated from the training data.

The class weights were calculated using the training split only:

`[0.2416, 7.8751, 3.0252, 27.3101, 2.7230]`

The purpose of weighted loss was to penalize mistakes on minority classes more heavily and investigate whether their recall could be improved.

---

## Validation results

Model selection was performed using the validation dataset.

### Traditional ML vs Deep Learning

| Model | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Dummy Classifier | 82.78% | 18.12% | 20.00% |
| Logistic Regression | 67.34% | 48.07% | 76.22% |
| Random Forest | 97.91% | 90.37% | 87.58% |
| CNN Plain | 98.48% | 91.78% | 88.89% |
| CNN Weighted | 95.36% | 82.12% | 91.22% |
| LSTM Plain | 98.99% | **94.43%** | 92.62% |
| LSTM Weighted | 97.05% | 86.60% | 92.59% |
| GRU Plain | **99.02%** | 94.34% | 93.35% |
| GRU Weighted | 97.14% | 86.93% | **93.59%** |

*All metrics are expressed as percentages.*

### Key observations

- **LSTM Plain achieved the highest Macro-F1**, at 94.43%.
- **GRU Plain achieved the highest accuracy**, at 99.02%.
- LSTM Plain and GRU Plain had nearly identical Macro-F1 scores.
- CNN Plain performed strongly, achieving 91.78% Macro-F1.
- Random Forest achieved 90.37% Macro-F1, making it a competitive traditional ML baseline.
- Dummy Classifier achieved 82.78% accuracy while completely failing to identify minority classes, demonstrating why accuracy alone is misleading.

The difference between LSTM Plain and GRU Plain in Macro-F1 was approximately 0.09 percentage points. This is too small to confidently establish a meaningful performance difference from a single validation run.

### Selected model

The Bidirectional LSTM Plain model was selected for final test evaluation because it achieved the highest validation Macro-F1 and demonstrated strong precision and recall across the five classes.

---

## Class-wise validation performance

### LSTM Plain

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| N | 99.28% | 99.71% | 99.49% | 10,871 |
| S | 94.54% | 83.18% | 88.50% | 333 |
| V | 97.02% | 97.58% | 97.30% | 868 |
| F | 91.95% | 83.33% | 87.43% | 96 |
| Q | 99.58% | 99.27% | 99.43% | 965 |

The model achieved strong performance on Normal, Ventricular, and Unknown heartbeat classes.

Supraventricular and Fusion beats remained comparatively more challenging, which is expected given their lower representation and similarities to other heartbeat morphologies.

### Validation confusion matrix

Rows represent true labels and columns represent predicted labels.

| True \ Predicted | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|
| N | 10,839 | 15 | 12 | 1 | 4 |
| S | 53 | 277 | 3 | 0 | 0 |
| V | 14 | 1 | 847 | 6 | 0 |
| F | 6 | 0 | 10 | 80 | 0 |
| Q | 6 | 0 | 1 | 0 | 958 |

The main validation errors involved Supraventricular beats being classified as Normal and Fusion beats being confused with Ventricular beats.

---

## Final test evaluation

The selected LSTM Plain model was evaluated on the separate Kaggle test dataset.

No model selection was performed using the test results.

### Test performance

| Metric | Score |
|---|---:|
| Accuracy | 98.70% |
| Macro Precision | 94.70% |
| Macro Recall | 91.30% |
| Macro-F1 | 92.89% |
| Balanced Accuracy | 91.30% |
| Weighted F1 | 98.67% |

### Class-wise test performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| N | 99.04% | 99.64% | 99.34% | 18,118 |
| S | 91.77% | 78.24% | 84.47% | 556 |
| V | 97.21% | 96.20% | 96.70% | 1,448 |
| F | 85.99% | 83.33% | 84.64% | 162 |
| Q | 99.50% | 99.07% | 99.28% | 1,608 |

### Test confusion matrix

| True \ Predicted | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|
| N | 18,052 | 36 | 21 | 2 | 7 |
| S | 111 | 435 | 8 | 2 | 0 |
| V | 33 | 3 | 1,393 | 18 | 1 |
| F | 18 | 0 | 9 | 135 | 0 |
| Q | 13 | 0 | 2 | 0 | 1,593 |

### Validation vs test performance

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 98.99% | 98.70% |
| Macro Precision | 96.48% | 94.70% |
| Macro Recall | 92.62% | 91.30% |
| Macro-F1 | 94.43% | 92.89% |
| Weighted F1 | 98.97% | 98.67% |

The test Macro-F1 was approximately 1.54 percentage points lower than the validation Macro-F1. The overall performance remained strong, although the model showed lower recall for Supraventricular beats.

The most frequent test errors included Supraventricular beats classified as Normal and Fusion beats classified as Normal or Ventricular.

These errors highlight the difficulty of distinguishing minority heartbeat categories with similar signal morphology.

---

## Plain vs weighted loss

| Model | Loss | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---|---:|---:|---:|
| CNN | Plain | 98.48% | 91.78% | 88.89% |
| CNN | Weighted | 95.36% | 82.12% | 91.22% |
| LSTM | Plain | 98.99% | 94.43% | 92.62% |
| LSTM | Weighted | 97.05% | 86.60% | 92.59% |
| GRU | Plain | 99.02% | 94.34% | 93.35% |
| GRU | Weighted | 97.14% | 86.93% | 93.59% |

### Effect of class weighting

Class weighting generally increased balanced accuracy or improved recall for selected minority classes, but reduced Macro-F1 and accuracy in these experiments.

For example:

- CNN weighted improved Fusion recall from 79.17% to 87.50%.
- LSTM weighted improved Fusion recall from 83.33% to 88.54%, but reduced Supraventricular recall slightly.
- GRU weighted improved Fusion recall from 82.29% to 87.50%, while Supraventricular recall decreased slightly.

The weighted models also produced more false-positive predictions for minority classes, reducing their precision.

These experiments demonstrate that class weighting is not guaranteed to improve overall minority-class performance. Its effectiveness depends on the trade-off between recall and precision.

---

## Error analysis

The test confusion matrix indicates that most classification errors occur between classes with fewer samples or similar signal characteristics.

Important error patterns include:

- Supraventricular beats misclassified as Normal.
- Fusion beats misclassified as Normal.
- Fusion beats misclassified as Ventricular.
- Occasional Ventricular beats misclassified as Normal.

These patterns motivate further investigation through waveform visualization and model explainability.

**Planned analysis:**

- Visualize representative misclassified test beats.
- Compare true and predicted classes.
- Investigate waveform morphology in commonly confused categories.
- Generate occlusion-based importance maps to study which temporal regions influence predictions.

---

## What I tried first, and what went wrong

### FAERS forecasting — abandoned initial approach

The first project idea involved forecasting FDA adverse-event report volumes using the openFDA API, focusing on adalimumab reports.

Exploratory analysis revealed:

- Large recurring January/February spikes.
- Report volumes during spikes substantially higher than typical periods.
- Level shifts in the time series.
- The previously observed spike pattern was no longer present after 2022.

These patterns made the dataset difficult to use for a clean forecasting experiment without further investigation into reporting behavior and underlying changes.

The forecasting project was abandoned in favor of ECG heartbeat classification, a more suitable sequence classification problem.

The original notebook is preserved in `notebooks/archive/` to document the initial exploration.

### First LSTM experiment — failed

The initial LSTM experiment used:

- A single-layer unidirectional LSTM.
- Classification using the final hidden state.
- Learning rate of 0.0001.
- Class-weighted loss.
- No gradient clipping.

The model failed to learn effectively and achieved approximately 0.29 validation Macro-F1.

Possible contributing factors included:

- The final hidden state had to carry information through the zero-padded portion of the signal.
- The learning rate was too low for effective training.
- Large class weights potentially destabilized optimization.

The architecture and training setup were revised to use a two-layer bidirectional LSTM, mean and max pooling over the sequence, a higher learning rate, and gradient clipping.

The revised model achieved 94.43% validation Macro-F1.

### Early CNN training mistake

An early CNN experiment unintentionally continued training from already-trained weights, inflating its apparent results.

The final experiments were rerun using freshly initialized models to ensure that the reported comparisons were based on independent training runs.

---

## Limitations

- **Potential patient leakage:** The available dataset does not provide patient identifiers, so patient-wise separation cannot be verified.
- **Limited dataset diversity:** The experiments use a single public dataset and recording setup.
- **Minority-class sample size:** Supraventricular and Fusion classes contain relatively few samples, making their metrics more sensitive to individual predictions.
- **Single-seed experiments:** Results were obtained using a fixed seed; multiple independent runs were not performed to estimate variability.
- **No external validation:** The models were not evaluated on another ECG dataset or independently collected recordings.
- **No clinical validation:** The models are educational experiments and are not suitable for medical diagnosis.
- **Validation support discrepancy:** Traditional ML reports and deep learning reports differ by one sample between the Fusion and Unknown validation classes. This likely originates from a data-loading or split inconsistency and should be resolved before treating small differences between models as definitive.

---

## Repository structure

```text
ecg-arrhythmia-classification/
│
├── config.yaml
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/                 # Dataset files, not committed
│   └── processed/            # Saved split indices
│
├── notebooks/
│   ├── 01_eda_and_ml_baselines.ipynb
│   ├── 02_pytorch_cnn.ipynb
│   ├── 03_pytorch_lstm_gru.ipynb
│   ├── 04_results_and_error_analysis.ipynb
│   ├── 05_final_test_evaluation.ipynb
│   └── archive/              # Initial FAERS forecasting attempt
│
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── baseline_models.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── utils.py
│
├── models/
│   └── final/
│
├── results/
│   ├── figures/
│   └── metrics/
│
└── app/
    └── gradio_app.py         # Deployment application
```

---

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/ashit0907/ecg-arrhythmia-classification.git

cd ecg-arrhythmia-classification
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) from Kaggle.

Place the dataset files inside:

```text
data/raw/
```

Required files:

```text
mitbih_train.csv
mitbih_test.csv
```

### 5. Run the notebooks

Execute the notebooks in order:

1. EDA and traditional ML baselines
2. CNN experiments
3. LSTM and GRU experiments
4. Model comparison and error analysis
5. Final test evaluation

The saved split indices ensure that the training and validation split can be reused consistently.

---

## Hardware and environment

| Component | Configuration |
|---|---|
| OS | Windows |
| GPU | NVIDIA RTX 4050 Laptop GPU |
| GPU VRAM | 6 GB |
| RAM | 24 GB |
| Framework | PyTorch |
| CUDA | CUDA-enabled PyTorch installation |

The experiments were conducted locally using GPU acceleration.

---

## Future work

- [ ] Visual error analysis of misclassified test beats.
- [ ] Occlusion-based explainability for ECG predictions.
- [ ] Interactive Gradio demo.
- [ ] Deployment on Hugging Face Spaces.
- [ ] Patient-wise evaluation using the original MIT-BIH recordings and `wfdb`.
- [ ] Multiple random seeds with mean and standard deviation.
- [ ] Evaluation on external ECG datasets.
- [ ] Investigate more advanced imbalance-handling techniques.

---

## References

1. Kachuee, M., Fazeli, S., & Sarrafzadeh, M. (2018). *ECG Heartbeat Classification: A Deep Transferable Representation.* [arXiv:1805.00794](https://arxiv.org/abs/1805.00794)

2. Moody, G. B., & Mark, R. G. (2001). *The impact of the MIT-BIH Arrhythmia Database.* [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)

3. [ECG Heartbeat Categorization Dataset — Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

---

## Author

**Ashit Nayak**

B.Tech Computer Science Engineering — AI & Machine Learning

[GitHub](https://github.com/ashit0907)
