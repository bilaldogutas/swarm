# database/db.py

from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Any
from datetime import datetime
import sqlite3

# Where the SQLite file will live (in the project root)
DB_PATH = Path("honeyhz_predictions.db")


@contextmanager
def get_connection():
    """
    Small helper to open/close a SQLite connection safely.
    Usage:
        with get_connection() as conn:
            ...
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """
    Create the predictions table if it doesn't already exist.
    Run once at app startup.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hive_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                swarm_risk REAL NOT NULL,
                risk_level TEXT NOT NULL
            );
            """
        )
        conn.commit()


def insert_prediction(
    hive_id: int,
    timestamp: datetime,
    swarm_risk: float,
    risk_level: str,
) -> None:
    """
    Save one prediction row to the database.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions (hive_id, timestamp, swarm_risk, risk_level)
            VALUES (?, ?, ?, ?);
            """,
            (hive_id, timestamp.isoformat(), swarm_risk, risk_level),
        )
        conn.commit()


def get_history_for_hive(hive_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch recent predictions for a given hive, newest first.
    Returns a list of dicts like:
        {"id": 1, "hive_id": 1, "timestamp": datetime(...), "swarm_risk": 0.5, "risk_level": "Watch"}
    """
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row  # allow dict-like access
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, hive_id, timestamp, swarm_risk, risk_level
            FROM predictions
            WHERE hive_id = ?
            ORDER BY timestamp DESC
            LIMIT ?;
            """,
            (hive_id, limit),
        )
        rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for row in rows:
        results.append(
            {
                "id": row["id"],
                "hive_id": row["hive_id"],
                "timestamp": datetime.fromisoformat(row["timestamp"]),
                "swarm_risk": row["swarm_risk"],
                "risk_level": row["risk_level"],
            }
        )
    return results
