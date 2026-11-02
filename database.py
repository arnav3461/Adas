import sqlite3
import os

class Database:
    def __init__(self, db_path="fleet_telematics.db"):
        self.db_path = db_path
        _init_db(self.db_path)

    def log_incident(self, timestamp, event_type, severity, description, video_clip_path=""):
        insert_incident(timestamp, event_type, severity, description, video_clip_path, db_path=self.db_path)

    def get_all_incidents(self):
        return get_all_incidents(db_path=self.db_path)

def _init_db(db_path="fleet_telematics.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            event_type TEXT,
            severity TEXT,
            description TEXT,
            video_clip_path TEXT,
            status TEXT DEFAULT 'unreviewed'
        )
    """)
    conn.commit()
    conn.close()

def insert_incident(timestamp, event_type=None, severity=None, description=None, video_clip_path="", db_path="fleet_telematics.db"):
    _init_db(db_path)
    status = "unreviewed"
    if isinstance(timestamp, dict):
        d = timestamp
        timestamp = d.get("trigger_timestamp_utc", d.get("timestamp", 0.0))
        event_type = d.get("event_type", "Anomaly")
        severity = d.get("severity_score", d.get("severity", "Medium"))
        description = d.get("description", "Safety incident")
        video_clip_path = d.get("video_clip_path", "")
        status = d.get("review_status", "unreviewed")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO incidents (timestamp, event_type, severity, description, video_clip_path, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, event_type, str(severity), description, video_clip_path, status))
    conn.commit()
    conn.close()

def get_all_incidents(db_path="fleet_telematics.db"):
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, event_type, severity, description, video_clip_path, status FROM incidents ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def update_incident_status(incident_id, status, db_path="fleet_telematics.db"):
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE incidents SET status = ? WHERE id = ?", (status, incident_id))
    conn.commit()
    conn.close()

def delete_incident(incident_id, db_path="fleet_telematics.db"):
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM incidents WHERE id = ?", (incident_id,))
    conn.commit()
    conn.close()

def clear_all_incidents(db_path="fleet_telematics.db"):
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM incidents")
    conn.commit()
    conn.close()