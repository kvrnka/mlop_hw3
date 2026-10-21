"""публикация модели и завершение обработки месяца."""

import logging
import os
from datetime import date
from pathlib import Path

from pipeline.db import advance_training_state, get_conn
from pipeline.retrain.evaluate_candidate import write_json_atomic
from pipeline.settings import CURRENT_DATASET, MODELS_DIR

logger = logging.getLogger(__name__)


def publish_model(evaluation: dict) -> dict:
    """Публикует только одобренного кандидата и фиксирует обработанный месяц.

    Читает решение evaluate_candidate, сохранённый CBM и pending-окно.
    При approved=True переносит CBM в каталог опубликованных моделей
    и обновляет latest.txt; при False оставляет активную модель прежней.
    В обоих случаях после сохранения и оценки кандидата фиксирует окно
    и training_state. Повтор завершённого месяца ничего не изменяет;
    при частичном сбое повтор использует уже перенесённые файлы.
    """
    version = evaluation["version"]
    approved = evaluation["approved"]
    batch_month = date.fromisoformat(evaluation["batch_month"])
    staging = Path(evaluation["staging_model"])
    final_model = MODELS_DIR / f"my_catboost_{version}.cbm"
    latest_path = MODELS_DIR / "latest.txt"

    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT next_batch_month FROM training_state WHERE id = 1 FOR UPDATE"
            )
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("training_state не инициализирован")
        next_month = row[0]

        if next_month > batch_month:
            logger.info("Месяц %s уже обработан; ничего не меняем", batch_month)
            return {"status": "already_processed", "version": version}

        active_version = latest_path.read_text(encoding="utf-8").strip()
        allowed_versions = {evaluation["current_version"]}
        if approved:
            allowed_versions.add(version)

        pending = Path(evaluation["dataset_path"])

        if approved:
            if staging.exists():
                os.replace(staging, final_model)
            elif not final_model.is_file():
                raise FileNotFoundError(f"Нет кандидата или опубликованной модели {version}")

            write_json_atomic(
                MODELS_DIR / f"my_catboost_{version}.metrics.json", evaluation
            )
            latest_tmp = MODELS_DIR / f".latest.{version}.txt"
            try:
                latest_tmp.write_text(version + "\n", encoding="utf-8")
                os.replace(latest_tmp, latest_path)
            finally:
                latest_tmp.unlink(missing_ok=True)
            active_version = version
            status = "published"
            logger.info("Опубликована модель %s", version)
        else:
            if not staging.is_file():
                raise FileNotFoundError(f"Не найден сохранённый кандидат {staging}")
            status = "rejected"
            logger.info("Кандидат %s отклонён: %s", version, evaluation["reason"])

        if evaluation["approved"]:
            model_path = (
                MODELS_DIR / f"my_catboost_{evaluation['version']}.cbm"
            )
        else:
            model_path = Path(evaluation["staging_model"])

        return {
            **evaluation,
            "status": status,
            "active_version": active_version,
            "model_path": str(model_path),
        }
    finally:
        conn.close()
