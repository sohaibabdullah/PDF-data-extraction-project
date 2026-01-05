import os
import shutil
import gspread
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION (KEPT EXACTLY AS REQUESTED) ---
DRIVE_DESTINATION_FOLDER_ID = os.getenv("DRIVE_DESTINATION_FOLDER_ID")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
DRIVE_LOGS_FOLDER_ID = os.getenv("DRIVE_LOGS_FOLDER_ID") 
LOCAL_LOGS_FOLDER = "Log_Output" 
LOCAL_SOURCE_FOLDER = "Formatted_Excel_Output"

# --- AUTHENTICATION & SETUP ---
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/spreadsheets'
]

def get_services():
    """Authenticates and returns the Drive service and Sheets client."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        print("Error: 'token.json' not found. Please run the setup script first.")
        return None, None
    
    # Build Drive Service
    drive_service = build('drive', 'v3', credentials=creds)
    # Build Sheets Client (gspread)
    sheet_client = gspread.authorize(creds)
    
    return drive_service, sheet_client

# --- HELPER FUNCTIONS ---
def zip_folder(folder_path):
    """Zips the folder and returns the path to the zip file."""
    try:
        # Saving the zip file in the same directory as the folder
        zip_path = shutil.make_archive(folder_path, 'zip', folder_path)
        return zip_path
    except Exception as e:
        print(f"   [!] Error zipping {folder_path}: {e}")
        return None

def create_drive_subfolder(service, folder_name, parent_id):
    """Creates a subfolder in Drive and returns its ID."""
    metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    try:
        file = service.files().create(body=metadata, fields='id').execute()
        return file.get('id')
    except Exception as e:
        print(f"   [!] Error creating Drive folder: {e}")
        return None

def upload_file_to_drive(service, file_path, folder_id):
    """Uploads a file to the specified Drive folder."""
    name = os.path.basename(file_path)
    metadata = {'name': name, 'parents': [folder_id]}
    media = MediaFileUpload(file_path, mimetype='application/zip', resumable=True) # mimetype applies generally
    
    try:
        file = service.files().create(body=metadata, media_body=media, fields='id').execute()
        return file.get('id')
    except Exception as e:
        print(f"   [!] Error uploading file: {e}")
        return None

def set_public_permission(service, file_id):
    """Sets permission to 'Anyone with link' (Viewer) and returns the link."""
    try:
        service.permissions().create(
            fileId=file_id,
            body={'type': 'anyone', 'role': 'reader'},
            fields='id'
        ).execute()
        file = service.files().get(fileId=file_id, fields='webViewLink').execute()
        return file.get('webViewLink')
    except Exception as e:
        print(f"   [!] Error setting permissions: {e}")
        return None

def update_sheet_row(client, sheet_id, folder_name, link, log_text, link_txt, link_xlsx):
    """Updates Sheet Columns C, I, J, K, P, Q, R."""
    try:
        sh = client.open_by_key(sheet_id)
        ws = sh.sheet1
        
        # Search Column C (Index 3)
        cell = ws.find(folder_name, in_column=3)
        
        if cell:
            row = cell.row
            
            # 1. COLOR COLUMN C (Blue #4285f4)
            ws.format(f"C{row}", {
                "backgroundColor": {"red": 0.259, "green": 0.522, "blue": 0.957}
            })

            # 2. UPDATE STATUS (Col I) - Force 'Done ' with space
            ws.update(
                range_name=f"I{row}", 
                values=[["Done "]], 
                value_input_option="USER_ENTERED"
            )

            # 3. UPDATE DATE/TIME (Col J) - Dhaka UTC+6
            dhaka_time = datetime.now(timezone.utc) + timedelta(hours=6)
            time_str = f"{dhaka_time.day} {dhaka_time.strftime('%b %y')}, {dhaka_time.strftime('%I').lstrip('0')}:{dhaka_time.strftime('%M %p')}"
            ws.update_cell(row, 10, time_str)

            # 4. UPDATE MAIN LINK (Col K)
            ws.update_cell(row, 11, link)

            # 5. UPDATE LOG TEXT (Col P)
            ws.update_cell(row, 16, log_text)

            # 6. UPDATE LOG TXT LINK (Col Q)
            ws.update_cell(row, 17, link_txt)

            # 7. UPDATE LOG XLSX LINK (Col R)
            ws.update_cell(row, 18, link_xlsx)
            
            return True, row
        else:
            return False, None
    except Exception as e:
        print(f"   [!] Sheet Error: {e}")
        return False, None

# --- PUBLIC FUNCTION TO CALL FROM MAIN PROJECT ---
def process_and_upload_folder(target_folder_name, log_message):
    """
    Main entry point to process a single folder.
    Args:
        target_folder_name (str): The name of the folder inside Formatted_Excel_Output (e.g., "Dhaka-1")
        log_message (str): The log text to write in Column P.
    """
    print(f"\n--- STARTING UPLOAD MODULE FOR: {target_folder_name} ---")
    
    # 1. Validate paths
    full_folder_path = os.path.join(LOCAL_SOURCE_FOLDER, target_folder_name)
    if not os.path.exists(full_folder_path):
        print(f"Error: Folder '{full_folder_path}' does not exist.")
        return False

    # 2. Setup Services
    drive_service, sheet_client = get_services()
    if not drive_service or not sheet_client:
        return False

    # 3. Process Steps
    
    # A. Zip
    zip_path = zip_folder(full_folder_path)
    if not zip_path: return False
    print("   -> Zipped.")

    # B. Drive: Create Folder
    new_drive_folder_id = create_drive_subfolder(drive_service, target_folder_name, DRIVE_DESTINATION_FOLDER_ID)
    if not new_drive_folder_id: return False
    print("   -> Drive folder created.")

    # C. Drive: Upload Zip
    uploaded_id = upload_file_to_drive(drive_service, zip_path, new_drive_folder_id)
    if not uploaded_id: return False
    print("   -> Zip uploaded.")

    # D. Drive: Share Main Zip
    share_link = set_public_permission(drive_service, new_drive_folder_id)
    if not share_link: return False
    print("   -> Permission set & Main Link generated.")

    # E. Process Log Files (TXT and XLSX)
    link_txt = ""
    link_xlsx = ""

    txt_filename = f"{target_folder_name}_processing_log.txt"
    xlsx_filename = f"{target_folder_name}_processing_report.xlsx"
    
    txt_path = os.path.join(LOCAL_LOGS_FOLDER, txt_filename)
    xlsx_path = os.path.join(LOCAL_LOGS_FOLDER, xlsx_filename)

    # Handle .txt Log
    if os.path.exists(txt_path):
        print(f"   -> Found Log TXT: {txt_filename}")
        txt_id = upload_file_to_drive(drive_service, txt_path, DRIVE_LOGS_FOLDER_ID)
        if txt_id:
            link_txt = set_public_permission(drive_service, txt_id)
    else:
        print(f"   -> [!] Log TXT not found: {txt_path}")

    # Handle .xlsx Report
    if os.path.exists(xlsx_path):
        print(f"   -> Found Log XLSX: {xlsx_filename}")
        xlsx_id = upload_file_to_drive(drive_service, xlsx_path, DRIVE_LOGS_FOLDER_ID)
        if xlsx_id:
            link_xlsx = set_public_permission(drive_service, xlsx_id)
    else:
        print(f"   -> [!] Log XLSX not found: {xlsx_path}")

    # F. Sheets: Update
    success, row_num = update_sheet_row(
        sheet_client, 
        GOOGLE_SHEET_ID, 
        target_folder_name, 
        share_link, 
        log_message, # Using the parameter passed from main program
        link_txt,
        link_xlsx
    )
    
    if success:
        print(f"   -> Sheet updated at Row {row_num}.")
    else:
        print(f"   -> WARNING: Could not find '{target_folder_name}' in the Google Sheet (Column C).")

    # Cleanup: Remove the local zip file
    try:
        os.remove(zip_path)
    except:
        pass
    
    print(f"--- FINISHED UPLOAD MODULE FOR: {target_folder_name} ---\n")
    return True

# --- DUMMY TEST BLOCK ---
# This allows you to test this file individually if needed
if __name__ == '__main__':
    # Test parameters
    TEST_FOLDER = "Dhaka-1" # Make sure this exists in Formatted_Excel_Output
    TEST_LOG = "Test Log Message from Module"
    
    process_and_upload_folder(TEST_FOLDER, TEST_LOG)