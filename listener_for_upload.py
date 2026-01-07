import OAuth_drive_upload_module
import logging
import sys
import time  # Imported for the sleep function
import traceback # Imported to print detailed errors if they happen
from pathlib import Path
from db import get_conn

# --- Logging Setup (Run once at startup) ---
script_dir = Path(__file__).parent
log_output_dir = script_dir / "OAuth_uopload_logs"
log_output_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_output_dir / f"upload_log.txt"

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

# --- Main Processing Function ---
def process_pending_uploads():
    """
    Connects to DB, checks for pending uploads, processes them, 
    and updates status.
    """
    conn = None
    try:
        # Establish a fresh connection for this cycle
        conn = get_conn()
        
        # Check for rows
        rows = conn.execute("""
            SELECT id, constituency_name, log_message, upload_status
            FROM logmessage
            WHERE upload_status = 'Not uploaded'
        """).fetchall()

        if not rows:
            logging.info("No pending uploads found. Waiting for next cycle.")
            return

        logging.info(f"Found {len(rows)} pending folder(s). Starting upload process...")

        for row in rows:
            folder_name = row[1]
            log_message = row[2]
            # row[3] is upload_status
            
            logging.info(f"Processing folder: {folder_name} with log message: {log_message}")
            
            try:
                # Attempt the upload
                uploaded = OAuth_drive_upload_module.process_and_upload_folder(
                    folder_name, 
                    log_message, 
                    existing_logger=logger
                )
                
                if uploaded:
                    upload_status = 'Uploaded'
                    conn.execute(
                        "UPDATE logmessage SET upload_status = ? WHERE constituency_name = ?",
                        (upload_status, row[1])
                    )
                    conn.commit()
                    logging.info(f"Successfully uploaded {folder_name}. Updated status to: {upload_status}")
                else:
                    logging.warning(f"Upload returned False for {folder_name}. Will retry next cycle.")
                    
            except Exception as e:
                logging.error(f"Error processing specific folder {folder_name}: {e}")
                # We continue to the next row even if this one failed

    except Exception as e:
        logging.error(f"Database or General Error in process loop: {e}")
        logging.error(traceback.format_exc())
    finally:
        # Always close the connection at the end of the cycle
        if conn:
            conn.close()

# --- Continuous Listener Loop ---
if __name__ == "__main__":
    logging.info("--- Upload Listener Service Started ---")
    
    while True:
        logging.info("Awake: Checking database for new rows...")
        
        process_pending_uploads()
        
        # Sleep for 30 minutes (30 * 60 seconds = 1800)
        logging.info("Sleeping for 30 minutes...")
        time.sleep(1800)