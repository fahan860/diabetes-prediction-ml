from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
TRAINED_MODEL_PATH = MODELS_DIR / "best_model.pkl"

TARGET_COLUMN = "Outcome"
ZERO_INVALID_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

TEST_SIZE = 0.2
RANDOM_STATE = 2
