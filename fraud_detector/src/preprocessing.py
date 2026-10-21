# Import standard libraries
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

CAT_COLS = [
    "merch", 
    "cat_id", 
    "name_1", 
    "name_2", 
    "gender",
    "street", 
    "one_city", 
    "us_state", 
    "post_code", 
    "jobs",
]

NUM_COLS = [
    "amount", 
    "lat", 
    "lon", 
    "population_city", 
    "merchant_lat", 
    "merchant_lon",
]

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """одинаковая предобработка для train и inference"""
    required = CAT_COLS + NUM_COLS + ["transaction_time"]
    # проверка, что все столбцы есть
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"в данных нет нужных столбцов: {missing}")

    X = df[CAT_COLS + NUM_COLS].copy()

    # заполняем пропуски в категориальных признаках
    for col in CAT_COLS:
        X[col] = X[col].astype("string").fillna("__MISSING__").astype(str)

    # меняем странные значения на nan
    for col in NUM_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    X[NUM_COLS] = X[NUM_COLS].replace([np.inf, -np.inf], np.nan)

    # преобразуем дату
    dt = pd.to_datetime(df["transaction_time"], errors="coerce")
    if not dt.notna().any():
        raise ValueError("не удалось распознать transaction_time")
    X["hour"] = dt.dt.hour.astype(float)
    X["day_of_week"] = dt.dt.dayofweek.astype(float)
    X["month"] = dt.dt.month.astype(float)

    return X

def run_preproc(df: pd.DataFrame) -> pd.DataFrame:
    """обертка над prepare_features для совместимости"""
    result = prepare_features(df)

    if "target" in df.columns:
        result["target"] = df["target"]

    return result
