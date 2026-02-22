from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE, ZERO_INVALID_COLUMNS


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataset.copy()
    cleaned[ZERO_INVALID_COLUMNS] = cleaned[ZERO_INVALID_COLUMNS].replace(0, np.nan)

    for column in ZERO_INVALID_COLUMNS:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    cleaned["Insulin"] = np.log1p(cleaned["Insulin"])
    return cleaned


def split_and_scale(dataset: pd.DataFrame):
    features = dataset.drop(TARGET_COLUMN, axis=1)
    target = dataset[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, x_train_scaled, x_test_scaled, scaler
