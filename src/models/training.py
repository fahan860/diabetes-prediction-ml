from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def build_models() -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "svm": SVC(kernel="rbf", class_weight="balanced", random_state=42, probability=True),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }


def train_models(models: Dict[str, object], x_train, y_train, x_train_scaled):
    models["logistic_regression"].fit(x_train_scaled, y_train)
    models["random_forest"].fit(x_train, y_train)
    models["svm"].fit(x_train_scaled, y_train)
    models["knn"].fit(x_train_scaled, y_train)
    return models


def evaluate_models(models: Dict[str, object], x_test, y_test, x_test_scaled):
    predictions = {
        "logistic_regression": models["logistic_regression"].predict(x_test_scaled),
        "random_forest": models["random_forest"].predict(x_test),
        "svm": models["svm"].predict(x_test_scaled),
        "knn": models["knn"].predict(x_test_scaled),
    }

    metrics = {}
    for name, y_pred in predictions.items():
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    return metrics
