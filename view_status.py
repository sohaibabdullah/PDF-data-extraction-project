import time
import sqlite3
import pandas as pd

DB = "status/logmessage.db"
OUT = "status/logmessage_view.csv"

while True:
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT constituency_name, log_message, upload_status FROM logmessage",
        conn
    )
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    conn.close()
    time.sleep(10)  # refresh every 10 seconds
