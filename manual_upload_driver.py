# Import the new module
import OAuth_drive_upload_module
import logging
import sys
from pathlib import Path
import sqlite3
from db import get_conn
import pandas as pd



# Set up logging
script_dir = Path(__file__).parent
log_output_dir = script_dir / "OAuth_uopload_logs"
log_output_dir.mkdir(parents=True, exist_ok=True)
log_file_path =  log_output_dir / f"upload_log.txt"

logger = logging.getLogger()

if logger.hasHandlers():
    logger.handlers.clear()

logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_formatter = logging.Formatter('%(message)s - %(asctime)s - %(levelname)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_formatter = logging.Formatter('%(message)s')
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)


conn = get_conn()
rows = conn.execute("""
    SELECT id, constituency_name, log_message, upload_status
    FROM logmessage
    WHERE upload_status = 'Not uploaded'
""").fetchall()


for row in rows:
    folder_name = row[1]
    log_message = row[2]
    upload_status = row[3]
    
    logging.info(f"Processing folder: {folder_name} with log message: {log_message}")
    uploaded = OAuth_drive_upload_module.process_and_upload_folder(folder_name, log_message,existing_logger=logger)
    if uploaded:
        upload_status = 'Uploaded'
        conn.execute(
            "UPDATE logmessage SET upload_status = ? WHERE constituency_name = ?",
            (upload_status, row[1])
        )
        conn.commit()
        logging.info(f"Successfully uploaded {folder_name}. Updated CSV upload status to: {upload_status}")

        