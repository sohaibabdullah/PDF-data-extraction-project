# Import the new module
import OAuth_drive_upload_module
import pandas as pd
import logging
import sys
from pathlib import Path




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

# Read the CSV file
df = pd.read_csv(
    "status/logmessage.csv",
    header=None,
    names=["Constituency_name", "Log", "upload_status"]
)

for _, row in df.iterrows():
    folder_name = row["Constituency_name"]
    log_message = row["Log"]
    upload_status = row["upload_status"]
    
    if upload_status != 'Not uploaded':
        logging.info(f"Skipping {folder_name} as it is already processed with status: {upload_status}")
        continue

    logging.info(f"Processing folder: {folder_name} with log message: {log_message}")
    uploaded = OAuth_drive_upload_module.process_and_upload_folder(folder_name, log_message,existing_logger=logger)
    if uploaded:
        upload_status = 'Uploaded'
        df.loc[df["Constituency_name"] == folder_name, "upload_status"] = upload_status
        df.to_csv("status/logmessage.csv", index=False, header=False)
        logging.info(f"Successfully uploaded {folder_name}. Updated CSV upload status to: {upload_status}")

