# Diabetes Prediction Project

## Problem
Diabetes screening is a binary classification problem where false negatives are costly.  
This project predicts diabetes status (`Outcome`) using the Pima Indians Diabetes dataset and compares multiple machine learning models.

## Solution
The codebase now uses a modular Python architecture:
- Data loading and preprocessing are isolated in `src/data/processing.py`
- Model creation, training, and evaluation are isolated in `src/models/training.py`
- Artifact persistence is isolated in `src/utils/io.py`
- Runtime configuration is centralized in `src/config.py`
- A single entry point (`main.py`) executes the full pipeline end-to-end

## Tech Stack
- Python 3.10+
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Joblib

## Architecture
```text
.
├── data/
│   └── raw/
│       └── diabetes.csv
├── models/                  # generated model artifacts (ignored by git)
├── notebooks/
│   └── projet_diabete.ipynb
├── screenshots/             # demo images
├── src/
│   ├── config.py
│   ├── main.py
│   ├── data/
│   │   └── processing.py
│   ├── models/
│   │   └── training.py
│   └── utils/
│       └── io.py
├── .gitignore
├── main.py                  # repository entry point
└── requirements.txt
```

## Results
The pipeline trains and evaluates:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Outputs generated after running the project:
- `models/metrics.json`: per-model metrics (accuracy, precision, recall, F1, confusion matrix)
- `models/best_model.pkl`: best model by F1 score

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Notes
- Core logic from the original notebook is preserved (same preprocessing and model family).
- Notebook remains available in `notebooks/` for experimentation.
