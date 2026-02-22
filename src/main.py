from __future__ import annotations

from src.config import DATA_PATH, METRICS_PATH, MODELS_DIR, TRAINED_MODEL_PATH
from src.data.processing import clean_dataset, load_dataset, split_and_scale
from src.models.training import build_models, evaluate_models, train_models
from src.utils.io import ensure_directory, save_metrics_json, save_model


def main() -> None:
    dataset = load_dataset(str(DATA_PATH))
    dataset = clean_dataset(dataset)

    x_train, x_test, y_train, y_test, x_train_scaled, x_test_scaled, _ = split_and_scale(dataset)

    models = build_models()
    models = train_models(models, x_train, y_train, x_train_scaled)
    metrics = evaluate_models(models, x_test, y_test, x_test_scaled)

    best_model_name = max(metrics, key=lambda model_name: metrics[model_name]["f1"])

    ensure_directory(MODELS_DIR)
    save_metrics_json(metrics, METRICS_PATH)
    save_model(models[best_model_name], TRAINED_MODEL_PATH)

    print("Training completed.")
    print(f"Best model (by F1): {best_model_name}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Best model saved to: {TRAINED_MODEL_PATH}")


if __name__ == "__main__":
    main()
