import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(LOGS_DIR, "predictions.db")

def get_connection() -> sqlite3.Connection:
    os.makedirs(LOGS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initialize the predictions table if it does not already exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                road_distance_km REAL NOT NULL,
                traffic_level TEXT NOT NULL,
                weather TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                preparation_time_min REAL NOT NULL,
                predicted_eta_mins REAL NOT NULL,
                delay_risk TEXT NOT NULL
            )
            """
        )
        conn.commit()

def insert_prediction(
    road_distance_km: float,
    traffic_level: str,
    weather: str,
    vehicle_type: str,
    preparation_time_min: float,
    predicted_eta_mins: float,
    delay_risk: str,
    timestamp: str = None
) -> int:
    """Insert a prediction record into the SQLite database and return its id."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                timestamp,
                road_distance_km,
                traffic_level,
                weather,
                vehicle_type,
                preparation_time_min,
                predicted_eta_mins,
                delay_risk
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                road_distance_km,
                traffic_level,
                weather,
                vehicle_type,
                preparation_time_min,
                predicted_eta_mins,
                delay_risk
            )
        )
        conn.commit()
        return cursor.lastrowid

def get_recent_predictions(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve the most recent predictions ordered by id descending."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                id,
                timestamp,
                road_distance_km,
                traffic_level,
                weather,
                vehicle_type,
                preparation_time_min,
                predicted_eta_mins,
                delay_risk
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at: {DB_PATH}")
