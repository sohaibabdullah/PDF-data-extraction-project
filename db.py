import sqlite3
from pathlib import Path

DB_PATH = Path("status/logmessage.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logmessage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            constituency_name TEXT,
            log_message TEXT,
            upload_status TEXT DEFAULT 'Not uploaded'
        )
    """)
    conn.commit()
    return conn
