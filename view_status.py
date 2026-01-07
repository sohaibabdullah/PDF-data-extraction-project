from db import get_conn
import pandas as pd
import time

conn = get_conn()


# --- Continuous console view + CSV update ---
last_seen_id = 0

print("Listening for new rows in the database. Press Ctrl+C to stop.")

try:
    while True:
        # Fetch all rows for CSV
        df = pd.read_sql_query(
            "SELECT id, constituency_name, log_message, upload_status FROM logmessage ORDER BY id ASC",
            conn
        )
        
        # Update CSV
        df.to_csv("status/logmessage.csv", index=False, encoding="utf-8-sig")
        
        # Print only new rows to console
        new_rows = df[df["id"] > last_seen_id]
        for _, row in new_rows.iterrows():
            print(f"[ID {row['id']}] {row['constituency_name']} | {row['log_message']} | {row['upload_status']}")
            last_seen_id = max(last_seen_id, row["id"])
        
        time.sleep(60)  # Wait for 60 seconds before checking again

except KeyboardInterrupt:
    print("\nStopped listening.")
