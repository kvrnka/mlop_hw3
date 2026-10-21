"""сравнение кандидата и действующей модели на одном test-месяце"""

import json
import logging
import os
from pathlib import Path

from pipeline.settings import LABEL, MODELS_DIR

logger = logging.getLogger(__name__)
THRESHOLD = 0.5


def write_json_atomic(path: Path, payload: dict) -> None:
    """Записывает полный JSON через временный файл в том же каталоге."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _calculate_metrics(model, features, target) -> dict:
    """Считает метрики на переданных строках с подготовкой как в скорере."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, roc_auc_score

    names = list(model.feature_names_)

    model_features = features.loc[:, names].copy()
    for index in model.get_cat_feature_indices():
        column = names[index]
        if not pd.api.types.is_numeric_dtype(model_features[column]):
            model_features[column] = model_features[column].astype(str)

    probability = model.predict_proba(model_features)[:, 1]

    prediction = (probability > THRESHOLD).astype(int)
    return {
        "roc_auc": float(roc_auc_score(target, probability)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
    }


def evaluate_candidate(train_info: dict) -> dict:
    """Сравнивает сохранённый CBM с моделью из latest.txt на одном test.
    
    Читает staging_model, test_proc_path, latest.txt и действующий CBM.
    Пишет MODELS_DIR/evaluations/<version>.json с метриками и решением.
    Не меняет модели, latest.txt, окно или training_state. При повторе
    возвращает уже сохранённое решение для этой версии и месяца.
    """
    version = train_info["version"]
    report_path = MODELS_DIR / "evaluations" / f"{version}.json"

    if report_path.exists():
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        logger.info("Используем сохранённое решение: %s", report_path)
        return saved

    import pandas as pd
    from catboost import CatBoostClassifier

    test = pd.read_parquet(train_info["test_proc_path"])
    target = test[LABEL]
    features = test.drop(columns=[LABEL])

    current_version = (MODELS_DIR / "latest.txt").read_text(encoding="utf-8").strip()
    if not current_version:
        raise ValueError("latest.txt не содержит действующей версии")

    candidate = CatBoostClassifier()
    candidate.load_model(train_info["staging_model"], format="cbm")
    current = CatBoostClassifier()
    current.load_model(str(MODELS_DIR / f"my_catboost_{current_version}.cbm"), format="cbm")

    candidate_metrics = _calculate_metrics(candidate, features, target)
    current_metrics = _calculate_metrics(current, features, target)
    degraded = [
        name for name in candidate_metrics
        if candidate_metrics[name] < current_metrics[name]
    ]
    improved = any(
        candidate_metrics[name] > current_metrics[name]
        for name in candidate_metrics
    )
    approved = not degraded and improved
    if degraded:
        reason = "Отклонена: ухудшились " + ", ".join(degraded)
    elif not improved:
        reason = "Отклонена: обе метрики равны метрикам действующей модели"
    else:
        reason = "Одобрена: ухудшений нет, хотя бы одна метрика улучшилась"

    result = {
        **train_info,
        "approved": approved,
        "reason": reason,
        "current_version": current_version,
        "candidate_metrics": candidate_metrics,
        "current_metrics": current_metrics,
        "threshold": THRESHOLD,
        "roc_auc": candidate_metrics["roc_auc"],
        "f1": candidate_metrics["f1"],
        "test_rows": int(len(test)),
        "evaluation_path": str(report_path),
    }
    write_json_atomic(report_path, result)
    logger.info(
        "Кандидат %s: approved=%s; %s; candidate=%s; current=%s; отчёт=%s",
        version, approved, reason, candidate_metrics, current_metrics, report_path,
    )
    return result
