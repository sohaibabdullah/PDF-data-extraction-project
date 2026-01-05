import os
import io
import shutil
import zipfile
import re
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

# --- CONFIGURATION ---
LINKS_FILE = "pdflinks.txt"
DOWNLOAD_FOLDER = "pdf-voterlist"
SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service():
    """Authenticates and returns the Drive service."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        print("Error: 'token.json' not found. Please run the setup script first.")
        return None
    return build('drive', 'v3', credentials=creds)

def extract_file_id(url):
    """Extracts File ID from Drive URL."""
    # Matches /d/ID/ or id=ID
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None

def read_links(file_path):
    """Reads links from file and returns IDs."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return []
    with open(file_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    ids = []
    for url in urls:
        fid = extract_file_id(url)
        if fid:
            ids.append(fid)
    return ids
'''
def download_file(service, file_id, output_path):
    """Downloads a file from Drive by ID."""
    try:
        # Get file metadata to know the name (optional, but good for logging)
        file_meta = service.files().get(fileId=file_id).execute()
        file_name = file_meta.get('name', 'downloaded_file.zip')
        
        print(f"   -> Downloading: {file_name} (ID: {file_id})...")
        
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(output_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            # Optional: Print progress
            # print(f"Download {int(status.progress() * 100)}%.")
            
        print("   -> Download Complete.")
        return True
    except HttpError as e:
        print(f"   [!] HTTP Error downloading {file_id}: {e}")
        return False
    except Exception as e:
        print(f"   [!] Error: {e}")
        return False
'''
def download_file(service, file_id, output_path):
    """Downloads a file and RETURNS THE REAL FILENAME from Drive."""
    try:
        # Get metadata to find the Real Name (e.g. নোয়াখালী-৪.zip)
        file_meta = service.files().get(fileId=file_id).execute()
        real_file_name = file_meta.get('name', f'downloaded_{file_id}.zip')
        
        print(f"   -> Downloading: {real_file_name} ...")
        
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(output_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            
        print("   -> Download Complete.")
        return True, real_file_name # Return the name found on Drive
    except HttpError as e:
        print(f"   [!] HTTP Error downloading {file_id}: {e}")
        return False, None
    except Exception as e:
        print(f"   [!] Error: {e}")
        return False, None
    
def clean_folder_names(root_folder):
    """
    Scans folders inside root_folder.
    Removes spaces around hyphens (e.g. 'Dhaka - 1' -> 'Dhaka-1').
    """
    print("--- Cleaning Folder Names ---")
    items = os.listdir(root_folder)
    
    for item in items:
        full_path = os.path.join(root_folder, item)
        
        if os.path.isdir(full_path):
            # Regex to replace "space-space", "space-", "-space" with just "-"
            # \s* matches zero or more spaces
            new_name = re.sub(r'\s*-\s*', '-', item)
            
            # Also strip leading/trailing spaces just in case
            new_name = new_name.strip()
            
            if new_name != item:
                new_path = os.path.join(root_folder, new_name)
                try:
                    os.rename(full_path, new_path)
                    print(f"   -> Renamed: '{item}'  ==>  '{new_name}'")
                except Exception as e:
                    print(f"   [!] Could not rename '{item}': {e}")

def main_downloader():
    print("--- STEP 1: DOWNLOADER STARTED ---")
    
    # 1. Setup
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)
    
    service = get_drive_service()
    if not service: return

    # 2. Get IDs
    file_ids = read_links(LINKS_FILE)
    print(f"Found {len(file_ids)} links to process.")

    # 3. Download & Unzip Loop
    for index, fid in enumerate(file_ids):
        print(f"\nProcessing Link {index + 1}/{len(file_ids)}...")
        
        # Temp path for the raw zip
        temp_zip_path = os.path.join(DOWNLOAD_FOLDER, f"temp_{index}.zip")
        
        # 1. DOWNLOAD
        success, real_filename = download_file(service, fid, temp_zip_path)
        
        if success and real_filename:
            # 2. CREATE FOLDER FROM ZIP NAME
            # Remove .zip extension to get "নোয়াখালী-৪"
            folder_name_base = os.path.splitext(real_filename)[0]
            
            # Create path: pdf-voterlist/নোয়াখালী-৪
            target_folder = os.path.join(DOWNLOAD_FOLDER, folder_name_base)
            
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            # 3. UNZIP DIRECTLY INTO THAT FOLDER
            try:
                print(f"   -> Unzipping into: {folder_name_base}")
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_folder)
                
            except zipfile.BadZipFile:
                print("   [!] Error: Invalid zip file.")
            except Exception as e:
                print(f"   [!] Unzip Error: {e}")
            
            # Cleanup Zip
            try: os.remove(temp_zip_path)
            except: pass
        else:
            print("   [!] Skipping unzip due to download failure.")

    # 4. Cleanup / Rename Folders
    clean_folder_names(DOWNLOAD_FOLDER)
    
    print("\n--- STEP 1: COMPLETED ---")

if __name__ == '__main__':
    main_downloader()