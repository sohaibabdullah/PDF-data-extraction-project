import os
import shutil
import gspread
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from datetime import datetime, timedelta, timezone

# --- CONFIGURATION (UPDATE THESE) ---
# The ID of the Google Drive folder where you want to upload the output
DRIVE_DESTINATION_FOLDER_ID = "1mNDyMwjpUadtFEpVRYI7UNAEOhDwV9ya"

# The ID of your Google Sheet
GOOGLE_SHEET_ID = "1vccZ3wZdMS-MPsvl1BBkvGA2xjZyMNyjGtIZLi2Iqdg"

# ID where logs go
DRIVE_LOGS_FOLDER_ID = "1T283gpqhYaR4dY6Zq8puPs7ftsMI2pVp" 

# Name of the local folder containing logs
LOCAL_LOGS_FOLDER = "Log_Output" 

# The local folder containing your sub-folders
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

# --- STEP 1: ZIPPING ---
def zip_folder(folder_path):
    """Zips the folder and returns the path to the zip file."""
    try:
        # Saving the zip file in the same directory as the folder
        zip_path = shutil.make_archive(folder_path, 'zip', folder_path)
        return zip_path
    except Exception as e:
        print(f"   [!] Error zipping {folder_path}: {e}")
        return None

# --- STEP 2: GOOGLE DRIVE ---
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
    """Uploads the zip file to the specified Drive folder."""
    name = os.path.basename(file_path)
    metadata = {'name': name, 'parents': [folder_id]}
    media = MediaFileUpload(file_path, mimetype='application/zip', resumable=True)
    
    try:
        file = service.files().create(body=metadata, media_body=media, fields='id').execute()
        return file.get('id')
    except Exception as e:
        print(f"   [!] Error uploading file: {e}")
        return None

def set_public_permission(service, file_id):
    """Sets permission to 'Anyone with link' (Viewer) and returns the link."""
    try:
        # Permission: Anyone, Reader
        service.permissions().create(
            fileId=file_id,
            body={'type': 'anyone', 'role': 'reader'},
            fields='id'
        ).execute()
        
        # Get the Web View Link
        file = service.files().get(fileId=file_id, fields='webViewLink').execute()
        return file.get('webViewLink')
    except Exception as e:
        print(f"   [!] Error setting permissions: {e}")
        return None

# --- STEP 3: GOOGLE SHEETS (UPDATED WITH COLOR FIX) ---
def update_sheet_row(client, sheet_id, folder_name, link, log_text, link_txt, link_xlsx):
    """
    Updates Sheet:
    1. Col C: Blue Background (#4285f4)
    2. Col J: Status 'Done ' (Green Chip)
    3. Col K: Date & Time (Dhaka UTC+6)
    4. Col L: Drive Link (Main Zip)
    5. Col Q: External Log Text
    6. Col R: Log TXT Link
    7. Col S: Log XLSX Link
    """
    try:
        sh = client.open_by_key(sheet_id)
        ws = sh.sheet1
        
        # Search Column C (Index 3)
        cell = ws.find(folder_name, in_column=3)
        
        if cell:
            row = cell.row
            
            # --- 1. COLOR COLUMN C (Blue #4285f4) ---
            ws.format(f"C{row}", {
                "backgroundColor": {"red": 0.259, "green": 0.522, "blue": 0.957}
            })

            # --- 2. UPDATE STATUS (Col J) ---
            ws.update(
                range_name=f"J{row}", 
                values=[["Done "]], 
                value_input_option="USER_ENTERED"
            )

            # --- 3. UPDATE DATE/TIME (Col K) ---
            dhaka_time = datetime.now(timezone.utc) + timedelta(hours=6)
            time_str = f"{dhaka_time.day} {dhaka_time.strftime('%b %y')}, {dhaka_time.strftime('%I').lstrip('0')}:{dhaka_time.strftime('%M %p')}"
            ws.update_cell(row, 11, time_str)

            # --- 4. UPDATE MAIN LINK (Col L) ---
            ws.update_cell(row, 12, link)

            # --- 5. UPDATE LOG TEXT (Col Q) ---
            ws.update_cell(row, 17, log_text)

            # --- 6. UPDATE LOG TXT LINK (Col R -> Index 18) ---
            ws.update_cell(row, 18, link_txt)

            # --- 7. UPDATE LOG XLSX LINK (Col S -> Index 19) ---
            ws.update_cell(row, 19, link_xlsx)
            
            return True, row
        else:
            return False, None
    except Exception as e:
        print(f"   [!] Sheet Error: {e}")
        return False, None

# --- MAIN WORKFLOW ---
def main():
    print("--- AUTOMATION STARTED ---")
    
    # 1. Setup
    drive_service, sheet_client = get_services()
    if not drive_service or not sheet_client:
        return

    # 2. Scan Directory
    if not os.path.exists(LOCAL_SOURCE_FOLDER):
        print(f"Error: Folder '{LOCAL_SOURCE_FOLDER}' not found.")
        return

    # Get only directories
    items = [i for i in os.listdir(LOCAL_SOURCE_FOLDER) 
             if os.path.isdir(os.path.join(LOCAL_SOURCE_FOLDER, i))]
    
    total = len(items)
    print(f"Found {total} folders to process in '{LOCAL_SOURCE_FOLDER}'.\n")

    for index, folder_name in enumerate(items):
        print(f"[{index+1}/{total}] Processing: {folder_name}...")
        
        full_folder_path = os.path.join(LOCAL_SOURCE_FOLDER, folder_name)
        
        # A. Zip
        zip_path = zip_folder(full_folder_path)
        if not zip_path: continue
        print("   -> Zipped.")

        # B. Drive: Create Folder
        new_drive_folder_id = create_drive_subfolder(drive_service, folder_name, DRIVE_DESTINATION_FOLDER_ID)
        if not new_drive_folder_id: continue
        print("   -> Drive folder created.")

        # C. Drive: Upload Zip
        uploaded_id = upload_file_to_drive(drive_service, zip_path, new_drive_folder_id)
        if not uploaded_id: continue
        print("   -> Zip uploaded.")

        # D. Drive: Share
        share_link = set_public_permission(drive_service, new_drive_folder_id)
        if not share_link: continue
        print("   -> Permission set & Link generated.")

        # F. Define the Log Parameter (Dummy for now, can be passed from outside later)
        current_log_message = "Process started via Automation Script v1.0"

        # ... [Previous code: Zipping, Creating Drive Folder, Uploading Zip, Sharing] ...
        
        # --- NEW LOGIC START: PROCESS LOG FILES ---
        link_txt = ""
        link_xlsx = ""

        # Construct file names based on folder name
        # Format: Dhaka-1_processing_log.txt
        txt_filename = f"{folder_name}_processing_log.txt"
        xlsx_filename = f"{folder_name}_processing_report.xlsx"
        
        txt_path = os.path.join(LOCAL_LOGS_FOLDER, txt_filename)
        xlsx_path = os.path.join(LOCAL_LOGS_FOLDER, xlsx_filename)

        # 1. Handle .txt Log
        if os.path.exists(txt_path):
            print(f"   -> Found Log TXT: {txt_filename}")
            # Upload to the LOGS specific folder
            txt_id = upload_file_to_drive(drive_service, txt_path, DRIVE_LOGS_FOLDER_ID)
            if txt_id:
                link_txt = set_public_permission(drive_service, txt_id)
        else:
            print(f"   -> [!] Log TXT not found: {txt_path}")

        # 2. Handle .xlsx Report
        if os.path.exists(xlsx_path):
            print(f"   -> Found Log XLSX: {xlsx_filename}")
            # Upload to the LOGS specific folder
            xlsx_id = upload_file_to_drive(drive_service, xlsx_path, DRIVE_LOGS_FOLDER_ID)
            if xlsx_id:
                link_xlsx = set_public_permission(drive_service, xlsx_id)
        else:
            print(f"   -> [!] Log XLSX not found: {xlsx_path}")
        # --- NEW LOGIC END ---

        # F. Define the Log Parameter
        current_log_message = "Automated Upload"

        # G. Sheets: Update (Passing the NEW link_txt and link_xlsx)
        success, row_num = update_sheet_row(
            sheet_client, 
            GOOGLE_SHEET_ID, 
            folder_name, 
            share_link, 
            current_log_message,
            link_txt,   # New Arg
            link_xlsx   # New Arg
        )
        
        if success:
            print(f"   -> Sheet updated at Row {row_num}.")
        else:
            print(f"   -> WARNING: Could not find '{folder_name}' in the Google Sheet (Column C).")

        # Cleanup: Remove the local zip file
        try:
            os.remove(zip_path)
        except:
            pass
        
        print("   -> Complete.\n")

    print("--- AUTOMATION FINISHED ---")

if __name__ == '__main__':
    main()