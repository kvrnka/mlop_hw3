"""Фиксация успешно обработанного окна независимо от одобрения модели."""

import logging
import os
from datetime import date
from pathlib import Path

from pipeline.db import advance_training_state, get_conn
from pipeline.settings import CURRENT_DATASET, MODELS_DIR

logger = logging.getLogger(__name__)


def finalize_training(publish_info: dict) -> dict:
    """Фиксирует окно и следующий месяц после успешной задачи публикации.

    Читает результат publish_model, сохранённый CBM, отчёт и pending-окно.
    Переносит pending в CURRENT_DATASET и обновляет training_state для
    исходного batch_month при статусах published и rejected.
    Активную модель не меняет. Повтор завершённого месяца ничего не пишет;
    при сбое после переноса окна использует существующий CURRENT_DATASET.
    Проверяется наличие файлов, а не состав месяцев в самом parquet.
    """
    version = publish_info["version"]
    batch_month = date.fromisoformat(publish_info["batch_month"])

    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT next_batch_month "
                "FROM training_state "
                "WHERE id = 1 FOR UPDATE"
            )
            row = cursor.fetchone()

        if row is None:
            raise RuntimeError("training_state не инициализирован")

        if row[0] > batch_month:
            logger.info(
                "Месяц %s уже завершён; повторная фиксация не нужна",
                batch_month,
            )
            return {
                **publish_info,
                "finalization_status": "already_processed",
            }

        if row[0] < batch_month:
            raise ValueError(
                f"Нельзя пропустить месяц {row[0]} ради {batch_month}"
            )

        status = publish_info["status"]

        if status not in {"published", "rejected"}:
            raise ValueError(
                f"Месяц ещё не завершён, но статус публикации: {status}"
            )

        if (status == "published") != (publish_info["approved"] is True):
            raise ValueError(
                "Статус публикации не соответствует решению проверки"
            )

        model_path = Path(publish_info["model_path"])
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Не найдена сохранённая модель: {model_path}"
            )

        if not Path(publish_info["evaluation_path"]).is_file():
            raise FileNotFoundError("Не найден отчёт оценки кандидата")

        expected_active = (
            version
            if status == "published"
            else publish_info["current_version"]
        )
        active_version = (
            MODELS_DIR / "latest.txt"
        ).read_text(encoding="utf-8").strip()

        if active_version != expected_active:
            raise RuntimeError(
                "Активная версия изменилась до завершения этого месяца"
            )

        pending = Path(publish_info["dataset_path"])

        if pending.exists():
            os.replace(pending, CURRENT_DATASET)
        elif not CURRENT_DATASET.exists():
            raise FileNotFoundError(
                "Нет ни нового окна обучения, ни current.parquet"
            )

        advance_training_state(conn, batch_month)

        logger.info(
            "Месяц %s завершён; активная модель %s",
            batch_month,
            active_version,
        )

        return {
            **publish_info,
            "finalization_status": "completed",
            "processed_month": batch_month.isoformat(),
        }

    finally:
        conn.close()