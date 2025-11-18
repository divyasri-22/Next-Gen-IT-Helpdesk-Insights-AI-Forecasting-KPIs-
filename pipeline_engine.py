# orchestrator/core/pipeline_engine.py

import uuid
import time
from datetime import datetime
from typing import Dict, Any, List

from .task_registry import task_registry
from .db import Database
from .logger import get_logger

logger = get_logger(__name__)
db = Database()


class PipelineEngine:

    def __init__(self, pipeline_name: str, config: Dict[str, Any]):
        self.pipeline_name = pipeline_name
        self.config = config
        self.tasks: List[Dict[str, Any]] = config.get("tasks", [])

    def run(self):
        run_id = str(uuid.uuid4())
        logger.info(f"Starting pipeline '{self.pipeline_name}' (run_id={run_id})")

        db.connect()
        start_time = datetime.utcnow().isoformat()

        # Insert pipeline run start record
        db.insert(
            """
            INSERT INTO pipeline_runs (pipeline_name, run_id, status, started_at)
            VALUES (?, ?, ?, ?);
            """,
            (self.pipeline_name, run_id, "running", start_time)
        )

        # Run tasks sequentially
        for task in self.tasks:
            task_name = task["name"]
            logger.info(f"Running task: {task_name}")

            task_start = datetime.utcnow().isoformat()

            try:
                func = task_registry.get(task_name)
                func(task.get("params", {}))  # run the task

                task_end = datetime.utcnow().isoformat()

                db.insert(
                    """
                    INSERT INTO task_runs (run_id, pipeline_name, task_name, status, started_at, finished_at)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (run_id, self.pipeline_name, task_name, "success", task_start, task_end)
                )

            except Exception as e:
                err = str(e)
                logger.error(f"Task '{task_name}' failed: {err}")

                task_end = datetime.utcnow().isoformat()

                db.insert(
                    """
                    INSERT INTO task_runs (run_id, pipeline_name, task_name, status, started_at, finished_at, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (run_id, self.pipeline_name, task_name, "failed", task_start, task_end, err)
                )

                # mark pipeline as failed
                db.insert(
                    """
                    UPDATE pipeline_runs SET status=?, finished_at=?, error_message=?
                    WHERE run_id=?;
                    """,
                    ("failed", task_end, err, run_id)
                )
                return  # stop pipeline

        # Pipeline success
        end_time = datetime.utcnow().isoformat()
        db.insert(
            """
            UPDATE pipeline_runs SET status=?, finished_at=?
            WHERE run_id=?;
            """,
            ("success", end_time, run_id)
        )

        logger.info(f"Pipeline '{self.pipeline_name}' finished successfully.")


