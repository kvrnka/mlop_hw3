"""Обучение CatBoost на новом трёхмесячном окне."""

import logging
from datetime import datetime, timezone
import os

import pandas as pd

from pipeline.settings import (
    CAT_FEATURES,
    LABEL,
    CANDIDATES_DIR,
    SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Отделяет target от признаков и приводит категориальные колонки к строкам.

    Категории совпадают с теми, что ожидает онлайн-скоринг после run_preproc.
    Возвращает матрицу признаков и вектор целевой переменной.
    """
    if LABEL not in frame.columns:
        raise ValueError(f"В датасете нет колонки {LABEL}")
    features = frame.drop(columns=[LABEL]).copy()
    for column in CAT_FEATURES:
        if column not in features.columns:
            raise ValueError(f"После препроцессинга нет категориального признака {column}")
        features[column] = features[column].astype(str)
    return features, frame[LABEL]


def fit_catboost(preproc_info: dict) -> dict:
    """Обучает CatBoost на train и оценивает его на свежем test-месяце.

    Читает обработанные train и test из parquet по путям из preproc_info.
    Обучает модель с фиксированным числом итераций и сохраняет кандидата
    в MODELS_DIR/candidates. Возвращает путь к модели, метрики и данные окна.

    Не изменяет latest.txt и указатель месяца в training_state.
    Повтор может создать дополнительный файл кандидата, но не изменяет
    действующую модель и исходные данные.
    """
    from catboost import CatBoostClassifier
    from sklearn.metrics import f1_score, roc_auc_score

    train = pd.read_parquet(preproc_info["train_proc_path"])
    test = pd.read_parquet(preproc_info["test_proc_path"])
    x_train, y_train = prepare_features(train)
    x_test, y_test = prepare_features(test)

    model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        allow_writing_files=False,
        verbose=20,
    )
    model.fit(
        x_train,
        y_train,
        cat_features=CAT_FEATURES,
    )

    # сохраняем кандидата до оценки на тесте
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    candidate_path = CANDIDATES_DIR / f"candidate_{version}.cbm"
    temp_path = candidate_path.with_suffix(".tmp")

    try:
        model.save_model(str(temp_path), format="cbm")
        os.replace(temp_path, candidate_path)
    finally:
        temp_path.unlink(missing_ok=True)

    logger.info("кандидат модели сохранён: %s", candidate_path)

    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba > SCORE_THRESHOLD).astype(int)

    f1 = float(f1_score(y_test, pred, zero_division=0))

    if y_test.nunique() < 2:
        roc_auc = None
        logger.warning("В test один класс: ROC-AUC не определён")
    else:
        roc_auc = float(roc_auc_score(y_test, proba))

    logger.info(
        "Кандидат %s: test_month=%s, ROC-AUC=%s, F1=%.4f",
        version,
        preproc_info["test_month"],
        roc_auc,
        f1,
    )

    return {
        "version": version,
        "staging_model": str(candidate_path),
        "train_proc_path": str(preproc_info["train_proc_path"]),
        "test_proc_path": str(preproc_info["test_proc_path"]),
        "dataset_path": preproc_info["dataset_path"],
        "batch_month": preproc_info["batch_month"],
        "train_months": preproc_info["train_months"],
        "test_month": preproc_info["test_month"],
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "roc_auc": roc_auc,
        "f1": f1,
    }
