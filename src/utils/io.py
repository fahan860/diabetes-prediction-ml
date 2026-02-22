from __future__ import annotations

import json
from pathlib import Path

import joblib


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_metrics_json(metrics: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_model(model, output_path: Path) -> None:
    joblib.dump(model, output_path)
