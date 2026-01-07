import os
from dotenv import load_dotenv
import serviceaccount_drive_upload_module    # <--- Reusing your existing success module
from datetime import datetime

# Load Config
load_dotenv()
DRIVE_ERROR_FOLDER_ID = os.getenv("DRIVE_ERROR_FOLDER_ID")
LOCAL_LOGS_FOLDER = "Log_Output"
LOCAL_SOURCE_FOLDER = "Formatted_Excel_Output"

def process_and_upload_error_folder(folder_name, log_message):
    """
    Handles folders that failed verification.
    1. Zips the folder.
    2. Creates a specific sub-folder in the 'Error Review' Drive.
    3. Uploads the Zip, the Log (.txt), and the Report (.xlsx) to that SAME folder.
    4. Creates a FAILURE_REASON.txt to explain why.
    """
    print(f"\n--- [ERROR MODULE] Processing: {folder_name} ---")
    
    if not DRIVE_ERROR_FOLDER_ID:
        print("❌ CRITICAL: DRIVE_ERROR_FOLDER_ID is missing in .env")
        return False

    # 1. Setup Services (Reuse function)
    drive_service, sheet_client = serviceaccount_drive_upload_module.get_services()
    if not drive_service: return False

    # 2. Get Path
    full_folder_path = os.path.join(LOCAL_SOURCE_FOLDER, folder_name)
    
    # 3. Zip (Reuse function)
    zip_path = serviceaccount_drive_upload_module.zip_folder(full_folder_path)
    if not zip_path: return False

    # 4. Create a specific Sub-folder for this error case
    # e.g., "Dhaka-1_REVIEW_REQUIRED"
    error_folder_name = f"{folder_name}_REVIEW_REQUIRED"
    drive_folder_id = serviceaccount_drive_upload_module.create_drive_subfolder(drive_service, error_folder_name, DRIVE_ERROR_FOLDER_ID)
    
    if not drive_folder_id:
        print("Failed to create error folder on Drive.")
        return False

    # 5. Upload The Zip File
    print(f"   -> Uploading Zip...")
    serviceaccount_drive_upload_module.upload_file_to_drive(drive_service, zip_path, drive_folder_id)

    # 6. Upload Logs (TXT/XLSX) to the SAME folder
    # This keeps everything together for easy debugging
    txt_filename = f"{folder_name}_processing_log.txt"
    xlsx_filename = f"{folder_name}_processing_report.xlsx"
    
    txt_path = os.path.join(LOCAL_LOGS_FOLDER, txt_filename)
    xlsx_path = os.path.join(LOCAL_LOGS_FOLDER, xlsx_filename)
    
    if os.path.exists(txt_path):
        print(f"   -> Uploading Log TXT...")
        serviceaccount_drive_upload_module.upload_file_to_drive(drive_service, txt_path, drive_folder_id)
        
    if os.path.exists(xlsx_path):
        print(f"   -> Uploading Log XLSX...")
        serviceaccount_drive_upload_module.upload_file_to_drive(drive_service, xlsx_path, drive_folder_id)
    # 7. Create & Upload FAILURE_REASON (Unique Name)
    # We add the folder name to the filename to prevent any conflicts
    # if the script runs in parallel or overlaps.
    reason_filename = f"{folder_name}_FAILURE_REASON.txt"
    
    with open(reason_filename, "w", encoding='utf-8') as f:
        f.write(f"Constituency: {folder_name}\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Failure Reason/Stats: {log_message}\n")
    
    # Upload it
    serviceaccount_drive_upload_module.upload_file_to_drive(drive_service, reason_filename, drive_folder_id)
    
    # Cleanup local unique file
    try:
        os.remove(reason_filename) 
    except:
        pass

    # 8. Get Share Link
    share_link = serviceaccount_drive_upload_module.set_public_permission(drive_service, drive_folder_id)
    
    # Cleanup Zip
    try:
        os.remove(zip_path)
    except:
        pass

    print(f"✅ Error Folder Uploaded. Review Link: {share_link}")
    print(f"--- [ERROR MODULE] Finished ---")
    
    return True