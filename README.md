# Multimodal Baseline Project

This repository provides a baseline multimodal learning pipeline for:

- **Task 1**: Question-specific personality trait regression  
- **Task 2**: User-level cognitive ability classification  

The system integrates **video, audio, and text** modalities and is designed as a straightforward, reproducible baseline for multimodal affective and behavioral analysis.

---

## Environment & Requirements

- Python **3.12.12**
- All required Python packages are listed in `requirements.txt`

### Environment Setup

It is recommended to use a dedicated Conda environment:

```bash
conda create -n multimodal_baseline python=3.12.12 -y
conda activate multimodal_baseline
pip install -r requirements.txt
```

---

## Project Pipeline Overview

The overall processing and training pipeline is as follows:

```
Video files
   ├─ extract_audios.py        → Audio (.wav)
   ├─ extract_features.py      → Multimodal features (.npy)
   ├─ train_model.py           → Task1 / Task2 trained models
   └─ val_model.py             → Validation results
```

Feature extraction and model training are performed at the **user level**, with question-specific aggregation depending on the task.

All scripts are designed to be executed sequentially.

---

## Project Structure

- `extract_audios.py`  
  Extracts audio tracks from raw video files.

- `extract_features.py`  
  Extracts and saves multimodal features (video, audio, text) as `.npy` files.

- `train_model.py`  
  Trains task-specific models for:
  - Task 1: Personality regression  
  - Task 2: Cognitive ability classification  

- `val_model.py`  
  Evaluates trained models on the validation set.

- `requirements.txt`  
  Python dependency list.

---

## Data Loading and Dataset Construction

Training and evaluation data are loaded using a custom `data_utils` module, which constructs datasets at the **user level**.

Only users with **complete multimodal features** are included.  
Specifically, a user is retained in the dataset **only if video, audio, and text features are available for all required questions**. Users with missing modality features are excluded to ensure consistent multimodal inputs.

### Task Definitions

- **Task 1 (Personality Regression)**  
  Task 1 is formulated as multiple independent regression problems.  
  Each question in `{q3, q4, q5, q6}` is treated as a separate task, where multimodal features from the corresponding question are used to predict a single personality trait score.

- **Task 2 (Cognitive Ability Classification)**  
  Task 2 uses multimodal features from all questions `{q1, q2, q3, q4, q5, q6}`.  
  For each user, features from different questions and modalities are concatenated to form a single representation, which is then used to predict the cognitive ability label.

All dataset construction logic, including user filtering and feature aggregation, is implemented in `data_utils.py`.

---

## Usage

### 1. Audio Extraction

Extract audio tracks from video files:

```bash
python extract_audios.py
```

> **Note:**  
> Input and output paths are defined inside the script.  
> Modify variables such as `BASE_DIR` or audio-related paths at the top of the file if necessary.

---

### 2. Multimodal Feature Extraction

Extract video, audio, and text features:

```bash
python extract_features.py
```

> **Note:**  
> Feature directories and data paths are currently **hard-coded** in the script.  
> Please edit variables such as `BASE_DIR`, `FEATURE_DIR`, and related paths to match your local setup.

All extracted features are saved in `.npy` format.

---

### 3. Model Training

Train models for both Task 1 and Task 2:

```bash
python train_model.py
```

This script:
- Loads pre-extracted multimodal features
- Trains task-specific models for Task 1 and Task 2
- Saves trained models to disk

> **Note:**  
> Training data paths, labels, and output directories are defined inside the script.

---

### 4. Validation

Evaluate trained models on the validation set:

```bash
python val_model.py
```

The validation script loads saved models and reports performance on the validation split.

> **Note:**  
> Paths for validation features and trained models are defined within the script.

---

## Outputs

- Trained models are saved under:
  ```
  ./trained_models/
  ```
- Feature files are saved as `.npy` arrays for efficient reuse.

---

## Notes

- This project is intended as a **baseline implementation**.
- All scripts currently rely on **hard-coded paths** for simplicity.
- For custom datasets or directory layouts, please update path variables at the top of each script accordingly.

---

## License

This project is provided for research and educational purposes.
