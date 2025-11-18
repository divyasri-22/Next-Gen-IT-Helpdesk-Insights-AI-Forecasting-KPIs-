from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Default DB location: <project_root>/db/orchestrator.db
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "orchestrator.db"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Return a SQLite connection. Creates the DB folder if needed.
    """
    path = Path(db_path) if db_path is not None else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: Optional[sqlite3.Connection] = None) -> None:
    """
    Initialize required tables if they do not exist.
    """
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    cursor = conn.cursor()

    # Table to track each pipeline run
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            pipeline_name TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            config_path TEXT,
            error_message TEXT
        )
        """
    )

    conn.commit()

    if close_after:
        conn.close()

    logger.info("Database initialized at %s", DB_PATH)


def start_pipeline_run(
    pipeline_name: str,
    run_id: str,
    started_at: str,
    config_path: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """
    Insert a new row for a pipeline run.

    Returns: database row id.
    """
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO pipeline_runs (
            run_id, pipeline_name, status, started_at, config_path
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, pipeline_name, "running", started_at, config_path),
    )
    conn.commit()
    row_id = cursor.lastrowid

    if close_after:
        conn.close()

    logger.info(
        "Started pipeline run: pipeline=%s run_id=%s row_id=%s",
        pipeline_name,
        run_id,
        row_id,
    )
    return row_id


def finish_pipeline_run(
    run_id: str,
    status: str,
    finished_at: str,
    error_message: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """
    Update an existing pipeline run with final status, finish time, and optional error.
    """
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE pipeline_runs
        SET status = ?, finished_at = ?, error_message = ?
        WHERE run_id = ?
        """,
        (status, finished_at, error_message, run_id),
    )
    conn.commit()

    if close_after:
        conn.close()

    logger.info(
        "Finished pipeline run: run_id=%s status=%s", run_id, status
    )


def get_recent_runs(
    limit: int = 50,
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    """
    Return recent pipeline runs as a list of dicts.
    """
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            id,
            run_id,
            pipeline_name,
            status,
            started_at,
            finished_at,
            config_path,
            error_message
        FROM pipeline_runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()
    results: List[Dict[str, Any]] = [dict(row) for row in rows]

    if close_after:
        conn.close()

    return results
