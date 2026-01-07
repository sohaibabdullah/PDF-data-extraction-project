############# ALL IMPORTS ################################
import os
import sys
import time
import json
import logging 
from pathlib import Path
from google.cloud import vision
from google.cloud import storage
from google.protobuf import json_format
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import freeze_support
import re
from collections import Counter
import pandas as pd
import threading
from dotenv import load_dotenv

from verify_output import run_verification 

import fitz  # PyMuPDF
from PIL import Image
import cv2
import numpy as np
import io
import matplotlib.pyplot as plt

import downloader
import serviceaccount_drive_upload_module
import error_upload_module

from db import get_conn

load_dotenv()

# --- GLOBAL REGION MAP ---
REGION_MAPPING = {}

def load_region_mapping():
    """Loads the CSV map into the global dictionary."""
    global REGION_MAPPING
    csv_path = "region_map.csv"
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Create a dictionary: Key = Constituency, Value = Region
            # cleanup string just in case
            df['Constituency'] = df['Constituency'].astype(str).str.strip()
            df['Region'] = df['Region'].astype(str).str.strip()
            
            REGION_MAPPING = pd.Series(df.Region.values, index=df.Constituency).to_dict()
            print(f"Loaded Region Map with {len(REGION_MAPPING)} entries.")
        except Exception as e:
            print(f"Error loading region_map.csv: {e}")
    else:
        print("Warning: region_map.csv not found. Region column will be empty.")
# Assume full_text is already available as a variable
# full_text = "your full text here"  # Replace with your variable


keys_patterns = {
    "অঞ্চল": r"^\s*[^\w]*\s*অঞ্চল\s*:?",
    
    "জেলা": r"^\s*[^\w]*\s*জেলা\s*:?",
    
    "উপজেলা/থানা": r"^\s*[^\w]*\s*উপজেলা\s*[/-]?\s*থানা\s*:?",
    
    "সিটি কর্পোরেশন/ পৌরসভা": r"^\s*[^\w]*\s*সিটি কর্পোরেশন\s*[/-]?\s*পৌরসভা\s*:?",
    
    "ইউনিয়ন/ওয়ার্ড/ক্যাঃ বোঃ": r"^\s*[^\w]*\s*ইউনিয়ন\s*[/-]?\s*ওয়ার্ড\s*[/-]?\s*ক্যাঃ?(?:\s|.)?বোঃ?\s*:?",
    
    "ওয়ার্ড নম্বর (ইউনিয়ন পরিষদের জন্য)": r"^\s*[^\w]*\s*ওয়ার্ড নম্বর\s*\(?ইউনিয়ন পরিষদের জন্য\)?\s*:?",
    
    "ডাকঘর": r"^\s*[^\w]*\s*ডাকঘর\s*:?",
    
    "পোষ্টকোড": r"^\s*[^\w]*\s*পো[ষস]্টকোড\s*:?",
    
    "ভোটার এলাকার নাম": r"^\s*[^\w]*\s*ভোটার এলাকার নাম\s*:?",
    
    "ভোটার এলাকার নম্বর": r"^\s*[^\w]*\s*ভোটার এলাকার নম্বর\s*:?"
}

def parse_header_text(text, patterns):
    """
    Parses header text using a line-by-line approach, and can now handle
    key-value pairs that are missing a colon.
    """
    #print(text)
    #print("_"*60)
    final_metadata = {key: "Not Found" for key in patterns.keys()}
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    keys_to_find = list(patterns.keys())

    def is_line_a_key(line_text):
        for pat in patterns.values():
            if re.search(pat, line_text):
                return True
        return False

    i = 0
    while i < len(lines):

        line = lines[i]

        matched_key = None
        
                 
        for key in keys_to_find:
            pattern = patterns[key]
            match = re.search(pattern, line)
            
            if match:
                matched_key = key
                # The value is everything in the line that comes AFTER the matched key pattern.
                value = line[match.end():].strip()
                # --- END OF 1st LOGIC ---

                # 2nd LOGIC: Look ahead to the next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    
                    # If the next line does NOT contain any known key pattern...
                    if not is_line_a_key(next_line):
                        # ...Append it to the current value
                        value = (value + " " + next_line).strip()
                        
                        # [IMPORTANT] Increment 'i' extra time to SKIP processing 
                        # the next line in the main loop (since we just merged it)
                        i += 1 

                #if value:
                final_metadata[key] = value
                #print(key,":",value)
                
                keys_to_find.remove(key)
                break 
        i += 1       

    #print ("printing metadata",final_metadata)
    return final_metadata

def find_header_box_coordinates(pil_image, search_area_ratio=0.4):
    """
    Uses OpenCV to find the y-coordinates of the top and bottom lines of the header box.
    Returns a tuple: (top_line_y, bottom_line_y).
    """
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    height, _ = gray.shape
    search_height = int(height * search_area_ratio)
    search_area = gray[0:search_height, :]
    inverted = cv2.bitwise_not(search_area)
    _, thresh = cv2.threshold(inverted, 128, 255, cv2.THRESH_BINARY)
    horizontal_projection = np.sum(thresh, axis=1)
    
    # Find the y-coordinate of the first line (the top of the box)
    first_line_y = np.argmax(horizontal_projection)
    
    # Erase the area around the first line
    erase_band_height = 20 # pixels
    start_erase = max(0, first_line_y - erase_band_height)
    end_erase = min(len(horizontal_projection), first_line_y + erase_band_height)
    horizontal_projection[start_erase:end_erase] = 0
    
    # Find the second line, which is the bottom of the box
    second_line_y = np.argmax(horizontal_projection)
    
    # Add a small padding to the coordinates to ensure we capture the lines themselves
    padding = 5 # pixels
    
    # Ensure the top coordinate is the smaller number
    top_y = min(first_line_y, second_line_y) - padding
    bottom_y = max(first_line_y, second_line_y) + padding

    return (max(0, top_y), bottom_y)

def extract_a_page(pdf_path, vision_client, pagenum):
    text_from_page= ""
    try:
        doc = fitz.open(pdf_path)
        if len(doc) >= 3:
            page = doc[pagenum] # Page 3 is at index 2
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            #top_y, bottom_y = find_header_box_coordinates(img)
            #crop_box = (0, top_y, img.width, bottom_y) 
            #header_image = img.crop(crop_box)
            #plt.imshow(header_image)
            #plt.show()
            #output_path = f"{pdf_file_name}.png"
            #header_image.save(output_path)

            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            image_content = img_byte_arr.getvalue()

            image = vision.Image(content=image_content)
            MAX_API_RETRIES = 200
            API_RETRY_DELAY_SECONDS = 10
            
            response = None 
            for attempt in range(MAX_API_RETRIES):
                try:
                    response = vision_client.text_detection(image=image)
                    break 
                except Exception as api_error:
                    logging.warning(f"Gender API call failed for '{pdf_file_name}' on attempt {attempt + 1}. Error: {api_error}")
                    if attempt < MAX_API_RETRIES - 1:
                        logging.info(f"Waiting for {API_RETRY_DELAY_SECONDS} seconds...")
                        time.sleep(API_RETRY_DELAY_SECONDS)
                    else:
                        errors.append(f"API call failed after {MAX_API_RETRIES} attempts.")
            
            if response and response.text_annotations:
                text_from_page = response.text_annotations[0].description
                #print(f"Text for gender extraction:\n {text_from_page}")
                doc.close()
                return text_from_page
        else:
            errors.append(f"PDF has fewer than {pagenum} pages.")
            doc.close()
            return ""
        
    except Exception as e:
        errors.append(f"Error during page {pagenum} processing: {e}")
        doc.close()
        return ""


def extract_metadata_robustly(pdf_path, vision_client, pagenum):
    """
    Extracts header metadata with high accuracy by parsing the filename and the
    dynamically cropped header of page 3, then cross-checking the results.
    """
    # This dictionary will store any errors encountered during the process
    if pagenum > 3:  # 0-indexed
        errors = [f"Could not extract header text within page-3 and 4."]
        return None, errors
    errors = []
    
    # --- 1. Parse metadata from the PDF filename ---
    pdf_file_name = pdf_path.name
    gender_from_name = "Not Found"
    if "female" in pdf_file_name.lower():
        gender_from_name = "মহিলা"
    elif "male" in pdf_file_name.lower():
        gender_from_name = "পুরুষ"
    elif "hijra" in pdf_file_name.lower():
        gender_from_name = "হিজড়া"
    
    if gender_from_name == "Not Found":
        logging.info("  [ERROR]Gender not found from filename, looking for gender from pages")
        text_from_page = extract_a_page(pdf_path, vision_client, pagenum)
        if re.search(r'মহিলা', text_from_page):
            gender_from_page = "মহিলা" 
        elif re.search(r'পুরুষ', text_from_page):
            gender_from_page = "পুরুষ"
        elif re.search(r'হিজড়া|হিজরা', text_from_page):
            gender_from_page = "হিজড়া"
        gender_from_name = gender_from_page
        logging.info(f"  [Resolved?]Gender found from page:{gender_from_page}")
    
    match = re.search(r'^(\d+)_', pdf_file_name)
    voter_area_number_from_name = match.group(1)[-4:] if match else ""

    # --- 2. Extract, DYNAMICALLY CROP, and OCR the header of Page 3 ---
    header_text_from_page3 = ""
    try:
        doc = fitz.open(pdf_path)
        if len(doc) >= 3:
            page = doc[pagenum] # Page 3 is at index 2
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            top_y, bottom_y = find_header_box_coordinates(img)
            crop_box = (0, top_y, img.width, bottom_y) 
            header_image = img.crop(crop_box)
            #plt.imshow(header_image)
            #plt.show()
            #output_path = f"{pdf_file_name}.png"
            #header_image.save(output_path)

            
            img_byte_arr = io.BytesIO()
            header_image.save(img_byte_arr, format='PNG')
            image_content = img_byte_arr.getvalue()

            image = vision.Image(content=image_content)
            MAX_API_RETRIES = 200
            API_RETRY_DELAY_SECONDS = 10
            
            response = None 
            for attempt in range(MAX_API_RETRIES):
                try:
                    response = vision_client.text_detection(image=image)
                    break 
                except Exception as api_error:
                    logging.warning(f"Metadata API call failed for '{pdf_file_name}' on attempt {attempt + 1}. Error: {api_error}")
                    if attempt < MAX_API_RETRIES - 1:
                        logging.info(f"Waiting for {API_RETRY_DELAY_SECONDS} seconds...")
                        time.sleep(API_RETRY_DELAY_SECONDS)
                    else:
                        errors.append(f"API call failed after {MAX_API_RETRIES} attempts.")
            
            if response and response.text_annotations:
                header_text_from_page3 = response.text_annotations[0].description

        else:
            errors.append(f"PDF has fewer than {pagenum} pages.")
        doc.close()
    except Exception as e:
        errors.append(f"Error during page {pagenum} processing: {e}")

    # --- 3. Parse Header Text and Build Final Metadata ---
    #print("I am here")
    if not header_text_from_page3:
        #print("I am inside to call page-4")
        errors.append(f"Could not extract any text from page {pagenum + 1} header. Trying next page.")
        metadata_from_header, recursive_errors = extract_metadata_robustly(pdf_path, vision_client, pagenum + 1)
        errors.extend(recursive_errors)
        if metadata_from_header is None:
            return None, errors
        else:
            return metadata_from_header, errors
        
            # Return immediately if we have no header text

    #logging.info(header_text_from_page3)
    #perfected parser
    metadata_from_header = parse_header_text(header_text_from_page3, keys_patterns)
    
    # Add the gender from the filename to the dictionary for use in the main process
    metadata_from_header["জেন্ডার"] = gender_from_name

    # --- 4. Cross-Check and Log Errors ---
    # Convert voter area number to Bangla for a fair comparison
    try:
        english_to_bangla_map = {'0':'০', '1':'১', '2':'২', '3':'৩', '4':'৪', '5':'৫', '6':'৬', '7':'৭', '8':'৮', '9':'৯'}
        bangla_voter_area_from_name = "".join([english_to_bangla_map[d] for d in voter_area_number_from_name])
    except KeyError:
        bangla_voter_area_from_name = "" # Handle cases with non-digits if any

    header_voter_area = metadata_from_header.get("ভোটার এলাকার নম্বর", "Not Found")
    
    if bangla_voter_area_from_name and (bangla_voter_area_from_name != header_voter_area):
        mismatch_error = (f"MISMATCH in Voter Area Number: Filename='{bangla_voter_area_from_name}', "
                          f"Header='{header_voter_area}'. Using filename value.")
        errors.append(mismatch_error)
        # We log this immediately so it appears next to the file being processed
        logging.warning(f"For '{pdf_file_name}': {mismatch_error}")
        metadata_from_header["ভোটার এলাকার নম্বর"] = bangla_voter_area_from_name
    elif bangla_voter_area_from_name:
        # If they match or header was empty, ensure the filename version is used
        metadata_from_header["ভোটার এলাকার নম্বর"] = bangla_voter_area_from_name


    # The function returns the final metadata AND a list of any errors found
    #print("PRINTING MEEETAdata:",metadata_from_header)
    return metadata_from_header, errors


#______________Process a Single PDF_______________________
def process_one_pdf(pdf_path,bucket_name):
    #print("This is the pdf path to be processed:",pdf_path)
    if not pdf_path.is_file():
        print(f"❌ ERROR: PDF file not found at '{pdf_path}'")
        return

    #print(f"--- Starting to process a single PDF: {pdf_path.name} ---")
    start_time = time.perf_counter()

    # Initialize the clients to talk to Google Cloud
    vision_client = vision.ImageAnnotatorClient()
    storage_client = storage.Client()

    # Define where the file will go in the cloud
    gcs_source_uri = f"gs://{bucket_name}/{pdf_path.name}"
    gcs_destination_uri = f"gs://{bucket_name}/{pdf_path.stem}_output/"

    # --- 1. Upload PDF to Google Cloud Storage ---
    #print(f"  - Step 1: Uploading to Cloud Storage bucket '{bucket_name}'...")
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(pdf_path.name)
    blob.upload_from_filename(str(pdf_path))
    #print("  - Upload complete.")

    # --- 2. Call the Vision API ---
    #print("  - Step 2: Calling the Vision API. This may take a few minutes...")
    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    gcs_source = vision.GcsSource(uri=gcs_source_uri)
    input_config = vision.InputConfig(gcs_source=gcs_source, mime_type='application/pdf')

    gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
    output_config = vision.OutputConfig(gcs_destination=gcs_destination, batch_size=100)

    async_request = vision.AsyncAnnotateFileRequest(
        features=[feature], input_config=input_config, output_config=output_config
    )

    operation = vision_client.async_batch_annotate_files(requests=[async_request])
    operation.result(timeout=420) # Wait up to 7 minutes for the job
    #print("  - Vision API job finished.")

    # --- 3. Get and Print the Results ---
    #print("\n--- OCR RESULTS ---")
    all_pages_text = ""
    blob_list = list(storage_client.list_blobs(bucket_name, prefix=f"{pdf_path.stem}_output/"))

    for blob in blob_list:
        json_string = blob.download_as_text()
        # Use the standard json library, which is more robust
        response_data = json.loads(json_string)

        # Loop through the pages in the JSON response
        for page_response in response_data['responses']:
            # Check if text was found on the page
            if 'fullTextAnnotation' in page_response:
                all_pages_text += page_response['fullTextAnnotation']['text']

    #print(all_pages_text)
    #print("\n--- END OF RESULTS ---")

    # --- 4. Clean Up Files ---
    #print("\n  - Step 3: Cleaning up temporary files from the bucket...")
    for blob_to_delete in blob_list:
        blob_to_delete.delete()
    bucket.blob(pdf_path.name).delete()
    #print("  - Cleanup complete.")

    end_time = time.perf_counter()
    #print(f"\n✅ SUCCESS! OCR finished in {end_time - start_time:.2f} seconds.")

    return all_pages_text



#___________________Extract all Key values from extracted text_______________________
def extract_all_values(full_text, keys_patterns):
    all_values = {key: [] for key in keys_patterns}

    for key, pattern in keys_patterns.items():
        if key == "জেন্ডার":
            matches = re.findall(pattern, full_text)
            all_values[key] = matches if matches else [""]
        else:
            matches = re.findall(pattern, full_text)
            all_values[key] = [match.strip() for match in matches if match.strip()]

    return all_values

#_____________________________Build metadata (header data)___________________
def build_metadata(all_values):
    metadata = {}
    for key, values in all_values.items():
        if values and key != "জেন্ডার":  # For keys with colon
            first_value = values[0] if values[0] else ""
            # Check if value contains part of another key or is empty/invalid
            if not first_value or any(other_key in first_value for other_key in keys_patterns if other_key != key):
                metadata[key] = ""
            else:
                metadata[key] = first_value
        elif key == "জেন্ডার":  # Special handling for gender
            if values and values[0]:
                gender_match = values[0]
                if "(মহিলা)" in gender_match:
                    metadata[key] = "মহিলা"
                elif "(পুরুষ)" in gender_match:
                    metadata[key] = "পুরুষ"
                else:
                    metadata[key] = ""
            else:
                metadata[key] = ""
    return metadata

#________________________CLEAN ALL RECORD PAGES____________________________
def clean_records(full_text):
    """
    Cleans the text by removing noise, including lines preceding 'কর্তন করা হয়েছে', 
    'মাইগ্রেট হয়েছে', 'কর্তন করা', or 'মাইগ্রেট' that contain digits (with possible spaces) 
    and end with a dot. Combines a number line with the following 'নাম:' line into a 
    single record entry with a dot separator (e.g., '১৩৩৮. নাম: সুমি').
    
    Args:
        full_text (str): The raw text to clean.
    
    Returns:
        list: List of cleaned lines.
    """
    # Step 1: Split into lines
    lines = [line.strip() for line in full_text.split('\n') if line.strip()]

    # Step 2: Find starting index with enhanced pattern
    start_index = None
    for i, line in enumerate(lines):
        if "ভোটার তালিকা প্রকাশের তারিখ:" in line:
            start_index = i + 1
            for j in range(i + 1, len(lines)):
                cleaned_line = re.sub(r'^[|\-\[\]]+\s*', '', lines[j]).strip()
                cleaned_line = re.sub(r'\s*-[^\s]*$', '', cleaned_line).strip()
                if re.search(r'^[০-৯]+\.?\s*নাম:', cleaned_line):
                    start_index = j
                    break
            break
    if start_index is None:
        for i, line in enumerate(lines):
            cleaned_line = re.sub(r'^[|\-\[\]]+\s*', '', lines[i]).strip()
            cleaned_line = re.sub(r'\s*-[^\s]*$', '', cleaned_line).strip()
            if re.search(r'^[০-৯]+\.?\s*নাম:', cleaned_line):
                start_index = i
                break
    if start_index is None:
        start_index = 0

    record_lines = lines[start_index:]

    # Step 3: Define noise patterns
    metadata_keys = [
        "অঞ্চল", "জেলা", "উপজেলা/থানা", "সিটি কর্পোরেশন/ পৌরসভা", "ইউনিয়ন/ওয়ার্ড/ক্যাঃ বোঃ",
        "ওয়ার্ড নম্বর (ইউনিয়ন পরিষদের জন্য)", "ডাকঘর", "পোষ্টকোড", "ভোটার এলাকার নাম",
        "ভোটার এলাকার নম্বর", "উপজেলা", "ইউনিয়ন", "ভোটার এলাকার কোড"
    ]
    standalone_noise = [
        "ফরম-১", "ছবি ছাড়া", "চূড়ান্ত ভোটার তালিকা", "প্রকাশের তারিখ:", "প্রকাশের তারিখ :",
        "কর্তন", "মাইগ্রেট", "কর্তন করা", "মাইগ্রেট হয়েছে", "কর্তন করা হয়েছে", "ডুপ্লিকেট","রেজিষ্ট্রেশন অফিসার",
        "চূড়ান্ত ভোটার","ফরম"
    ]
    noise_pattern = r"|".join(
        [re.escape(key) + r"\s*:" for key in metadata_keys] +
        [re.escape(noise) for noise in standalone_noise]
    )
    noise_pattern = r"(?:" + noise_pattern + r")"

    # Step 4: Filter noise and handle awkward signs
    cleaned_lines = []
    i = 0
    while i < len(record_lines):
        line = record_lines[i]
        # Remove leading symbols and trailing noise
        cleaned_line = re.sub(r'^[|\-\[\]]+\s*', '', line).strip()
        cleaned_line = re.sub(r'\s*-[^\s]*$', '', cleaned_line).strip()

        # Check for noise
        is_noise = re.search(noise_pattern, cleaned_line)

        if is_noise:
            if "কর্তন করা হয়েছে" in cleaned_line or "মাইগ্রেট হয়েছে" in cleaned_line or \
               "কর্তন করা" in cleaned_line or "মাইগ্রেট" in cleaned_line or "ডুপ্লিকেট" in cleaned_line:
                # Check if the last added line is a number (with possible spaces) ending with a dot
                if cleaned_lines and re.match(r'^[০-৯0-9]+(?:\s+[০-৯0-9]+)*\.$', cleaned_lines[-1]):
                    cleaned_lines.pop()  # Remove the previous number line
            i += 1
            continue  # Skip adding the noise line
        
        # Check if this line is a number that should combine with the next 'নাম:' line
        if re.match(r'^[০-৯0-9]+(?:\s+[০-৯0-9]+)*\.?$', cleaned_line):
            if i + 1 < len(record_lines):
                next_line = record_lines[i + 1]
                cleaned_next_line = re.sub(r'^[|\-\[\]]+\s*', '', next_line).strip()
                cleaned_next_line = re.sub(r'\s*-[^\s]*$', '', cleaned_next_line).strip()
                if re.search(r'\s*নাম:', cleaned_next_line):
                    # Combine the number and name lines, preserving or adding a dot
                    if cleaned_line.endswith('.'):
                        combined_line = f"{cleaned_line} {cleaned_next_line}"
                    else:
                        combined_line = f"{cleaned_line}. {cleaned_next_line}"
                    cleaned_lines.append(combined_line)
                    i += 2  # Skip both the number and name lines
                    continue
        
        # Clean trailing numbers from potential address lines
        if not re.match(r'^[০-৯]+\.?\s*নাম:', cleaned_line) and not is_noise:
            cleaned_line = re.sub(r'\s*[০-৯0-9]+\s*$', '', cleaned_line).strip()
        
        # Include the line if it's not noise or starts a record
        if cleaned_line or re.search(r'^[০-৯]+\.?\s*নাম:', cleaned_line):
            cleaned_lines.append(cleaned_line)
        i += 1
        
       

    # Step 5: Verify (uncomment for debugging)
    # print(f"Original lines: {len(lines)}")
    # print(f"Record lines: {len(record_lines)}")
    # print(f"Cleaned lines: {len(cleaned_lines)}")
    # print("First 20 cleaned lines:", cleaned_lines[:20])
    
    return cleaned_lines

   

#_______________________Extract ALL Records_________________________________
def extract_all_records(cleaned_lines):
    all_records = []
    current_record = None

    # Loop through each line of text from your list
    for line in cleaned_lines:
        line = line.strip()
        if not line:
            continue

        # --- Step 1: Identify the start of a new record ---
        # The pattern looks for a line starting with digits, an optional dot,
        # spaces, and then "নাম:".
        # It captures the serial number and the name.
        name_match = re.match(r'^([০-৯]+)\.?\s*নাম:\s*(.*)', line)
        if name_match:
            # If we were working on a previous record, save it before starting a new one.
            if current_record:
                all_records.append(current_record)

            # Create a new dictionary for the new person
            serial_number = name_match.group(1).strip()
            name = name_match.group(2).strip()
            current_record = {
                "সিরিয়াল নম্বর": serial_number,
                "নাম": name,
                "ভোটার নং": "",
                "পিতা": "",
                "মাতা": "",
                "পেশা": "",
                "জন্ম তারিখ": "",
                "ঠিকানা": ""
            }
            continue

        # If we are not at a "নাম:" line, we must be inside a current record.
        if current_record:
            # --- Step 2: Extract key-value pairs for the current record ---
            if line.startswith("ভোটার নং:"):
                current_record["ভোটার নং"] = line.split(":", 1)[1].strip()

            elif line.startswith("পিতা:"):
                current_record["পিতা"] = line.split(":", 1)[1].strip()

            elif line.startswith("মাতা:"):
                current_record["মাতা"] = line.split(":", 1)[1].strip()

            elif line.startswith("ঠিকানা:"):
                # Handle addresses that might span multiple lines
                # If the address field is already filled, we append the new line.
                if current_record["ঠিকানা"]:
                    current_record["ঠিকানা"] += ", " + line.split(":", 1)[1].strip()
                else:
                    current_record["ঠিকানা"] = line.split(":", 1)[1].strip()

            elif line.startswith("পেশা:"):
                # --- Step 3: Handle the special merged "পেশা" and "জন্ম তারিখ" line ---
                value_part = line.split(":", 1)[1].strip()

                if ',জন্ম তারিখ:' in value_part:
                    parts = value_part.split(',জন্ম তারিখ:')
                    current_record["পেশা"] = parts[0].strip()
                    current_record["জন্ম তারিখ"] = parts[1].strip()
                else:
                    # If "জন্ম তারিখ" is not in the line, the whole value is the profession
                    current_record["পেশা"] = value_part

            else:
                # If a line does not match any key (e.g., a continuation of an address),
                # append it to the 'ঠিকানা' field.
                if current_record["ঠিকানা"]: # Only append if address has been started
                    current_record["ঠিকানা"] += ", " + line

    # After the loop finishes, make sure to save the very last record
    if current_record:
        all_records.append(current_record)

    return all_records

#################NEW NEW NEW RETURN ALL RECORDS___________________________
def find_and_parse_records(full_text,metadata):
    """
    Parses voter records from raw text based on the user-provided robust algorithm.
    """
    all_records = []
    
    # Regex to find the start of a record. It's flexible:
    # - Handles junk characters `[|\s\[\]-]*` at the start.
    # - Captures the serial number `([০-৯]+)`.
    # - Handles the dot or dash `[\.-]`.
    # - Allows "নাম:" to be on the same line or the next line `(?:\s*\n)?\s*নাম:`.
    start_pattern = re.compile(r'[|\s\[\]-]*([০-৯]+)[\.-]\s*(?:\n)?[|\s\[\]-]*\s*নাম:')
    
    # Find all starting positions of valid records
    start_matches = list(start_pattern.finditer(full_text))
    
    #print(f"--- Found {len(start_matches)} record start points. ---")

    # Loop through the matches to define the "chunk" for each record
    for i, current_match in enumerate(start_matches):
        start_pos = current_match.start()
        
        # The chunk ends where the next record starts, or at the end of the text
        end_pos = start_matches[i + 1].start() if i + 1 < len(start_matches) else len(full_text)
        
        # Isolate the clean block of text for this single record
        record_block = full_text[start_pos:end_pos]
        
        # --- Now we parse ONLY inside this clean block ---
        
        # 1. Extract Serial and Name from the start of the block
        serial = current_match.group(1).strip()
        name_match = re.search(r'নাম:\s*(.*)', record_block, re.DOTALL)
        name_text = name_match.group(1).strip() if name_match else ""
        
        # The name might have other fields in it, so we clean it
        first_line_of_name = name_text.split('\n')[0]
        
        record = {
            "সিরিয়াল নম্বর": serial,
            "নাম": first_line_of_name,
        }

        # 2. Flexibly extract other known keys
        keys_to_find = ["ভোটার নং", "পিতা", "মাতা", "পেশা", "জন্ম তারিখ"]
        for key in keys_to_find:
            # Pattern: Optional junk, the key, colon, then capture the value on that line
            pattern = re.search(rf"[|\s\[\]-]*{key}:\s*(.*)", record_block)
            if pattern:
                record[key] = pattern.group(1).strip(' ,')
            else:
                record[key] = "" # Ensure key exists even if not found

        # 3. Handle the Address with your "Smart Stop" logic
        address_match = re.search(r"[|\s\[\]-]*ঠিকানা:\s*([\s\S]*)", record_block, re.DOTALL)
        if address_match:
            # Capture the entire potential address block first
            potential_address = address_match.group(1).strip()
            address_lines = potential_address.split('\n')
            
            final_address_parts = []
            address_ended = False
            
            # Get the district name from the metadata to use as a stopword
            district_name = metadata.get("জেলা")

            for line in address_lines:
                clean_line = line.strip()
                if not clean_line: continue

                # THE FIX IS HERE: Check if the district name exists in the line
                if district_name and district_name in clean_line:
                    # Find the position where the district name ends
                    end_pos = clean_line.find(district_name) + len(district_name)
                    # Truncate the line to keep everything up to that point
                    final_part = clean_line[:end_pos]
                    final_address_parts.append(final_part)
                    
                    # This is our hard stop condition
                    address_ended = True
                    break # Exit the loop immediately
                
                # If it's not a stop line, add the whole line
                final_address_parts.append(clean_line)
            
            # Join the collected parts into a single string
            full_address = ", ".join(final_address_parts)
            
            # Apply the fallback logic if needed
            if not address_ended and len(address_lines) > 1:
                # If district was never found in a multi-line address, take only the first line
                record["ঠিকানা"] = address_lines[0].strip(' ,')
            else:
                record["ঠিকানা"] = full_address.strip(' ,')
        else:
            record["ঠিকানা"] = ""

        # 4. Re-process the combined "পেশা" and "জন্ম তারিখ" field
        if record["পেশা"] and 'জন্ম তারিখ' in record["পেশা"]:
            value_part = record["পেশা"]
            if ',জন্ম তারিখ:' in value_part:
                parts = value_part.split(',জন্ম তারিখ:')
                record['পেশা'] = parts[0].strip()
                record['জন্ম তারিখ'] = parts[1].strip()
        
        all_records.append(record)

    return all_records

###_______________________Excel Writing Functio__________________________
def write_to_excel(metadata, all_records, constituency_name, base_output_dir, pdf_file_name,pdf_to_excel_map, map_lock):
    """
    Writes the metadata and voter records to a formatted Excel file with a
    dynamic path and filename.

    Args:
        metadata (dict): The dictionary of 11 header key-value pairs.
        all_records (list): A list of dictionaries, each a voter record.
        constituency_name (str): The name of the root constituency folder.
        base_output_dir (Path): The root directory where the output structure will be created.
    """
    #print("  - Preparing to write Excel file...")

    try:
        # --- 1. Sanitize folder and file names ---
        # Get values for path and filename, providing a default if a key is missing
        district = metadata.get("জেলা", "unknown_district")
        upazila = metadata.get("উপজেলা/থানা", "unknown_upazila")
        union = metadata.get("ইউনিয়ন/ওয়ার্ড/ক্যাঃ বোঃ", "unknown_union")
        voter_area = metadata.get("ভোটার এলাকার নাম", "unknown_area")
        gender = metadata.get("জেন্ডার", "unknown_gender")
        voter_area_number = metadata.get("ভোটার এলাকার নম্বর", "unknown_number")

        voter_area_full = voter_area
        # Clean the strings to make them valid folder/file names
        # This removes characters that are not allowed in Windows paths.
        def sanitize(name):
            return re.sub(r'[\\/*?:"<>|]', "", name).strip()

        # --- 2. Construct the dynamic path and filename ---
        MAX_PATH = 240
        def build():
            final_folder_name = f"{sanitize(voter_area)}_{sanitize(voter_area_number)}"
            output_path = (base_output_dir / sanitize(district) /
                        sanitize(upazila) / sanitize(union) / final_folder_name)

            filename = f"{sanitize(constituency_name)}_{sanitize(union)}_{sanitize(voter_area)}-{sanitize(voter_area_number)}_{sanitize(gender)}.xlsx"
            full_filepath = output_path / filename
            return output_path, full_filepath
        
        output_path, full_filepath = build()

        path_len = len(str(full_filepath))
        #print(path_len)
        def shorten(text, excess, min_len=10):
            return text[:max(len(text) - excess, min_len)].rstrip()
        if path_len > MAX_PATH:
            excess = path_len - MAX_PATH
            voter_area = shorten(sanitize(voter_area), excess//2)
            output_path, full_filepath = build()

        filename = full_filepath.name

        # --- 3. Create the directory structure ---
        # The `parents=True, exist_ok=True` part is crucial. It creates all
        # necessary parent folders and doesn't raise an error if they already exist.
        output_path.mkdir(parents=True, exist_ok=True)

        # --- 4. Prepare data for Excel writing ---
        # Create the header rows as a list of lists
        mapped_region = REGION_MAPPING.get(constituency_name.strip(), metadata.get("অঞ্চল", ""))

        header_rows = [
            ["আসন নাম", constituency_name],
            ["অঞ্চল", mapped_region],
            ["জেলা", district],
            ["উপজেলা/থানা", upazila],
            ["সিটি কর্পোরেশন/পৌরসভা", metadata.get("সিটি কর্পোরেশন/ পৌরসভা", "")],
            ["ইউনিয়ন/ওয়ার্ড/ক্যাঃ বোঃ", union],
            ["ওয়ার্ড নম্বর (ইউনিয়ন পরিষদের জন্য)", metadata.get("ওয়ার্ড নম্বর (ইউনিয়ন পরিষদের জন্য)", "")],
            ["ডাকঘর", metadata.get("ডাকঘর", "")],
            ["পোস্টকোড", metadata.get("পোষ্টকোড", "")],
            ["ভোটার এলাকার নাম", voter_area_full ],
            ["ভোটার এলাকার নম্বর", metadata.get("ভোটার এলাকার নম্বর", "")],
            ["ভোটার তালিকা", gender]
        ]

        # Create a pandas DataFrame for the header
        header_df = pd.DataFrame(header_rows)

        # Create a pandas DataFrame for the voter records
        records_df = pd.DataFrame(all_records)

        # --- 5. Write to the Excel file ---
        with pd.ExcelWriter(full_filepath, engine='openpyxl') as writer:
            # Write the header without its own column titles or index
            header_df.to_excel(writer, sheet_name='VoterList', startrow=0, header=False, index=False)

            # Write the voter records starting on a later row, with column titles
            records_df.to_excel(writer, sheet_name='VoterList', startrow=len(header_rows) , header=True, index=False)

        logging.info(f"  [SUCCESS] Excel file written successfully to: {filename} \n \tfor PDF: {pdf_file_name}")
        with map_lock:
            pdf_to_excel_map[pdf_file_name] = filename

    except Exception as e:
        print(f"  [ERROR] Failed to write Excel file. Reason: {e}")


#_______________________ Process Both Male and Female PDFs_________________
def process_malefemale_pdfs(pdf_pair, bucket_name,constituency_name, base_output_dir,pdf_to_excel_map, map_lock, critical_metadata_failures):

    """
    This function is the "work" that each parallel worker will do.
    It takes a pair of PDFs and processes both of them sequentially.
    """
    all_metadata_errors = []

    MAX_RETRIES = 10
    RETRY_DELAY_SECONDS = 10

    successfully_processed_pdfs = []

    for pdf_file in pdf_pair:
        processed_successfully = False
        #loop for retries
        for attempt in range(MAX_RETRIES):
            try:
                logging.info(f"--Processing '{pdf_file.name}'. Attempt {attempt + 1} of {MAX_RETRIES}.")
                #print(f"Processing '{pdf_file.name}'. Attempt {attempt + 1} of {MAX_RETRIES}")
                if 'vision_client' not in locals():
                    vision_client = vision.ImageAnnotatorClient()


                # Get VERIFIED metadata using the new algorithm implemented in robust function ---
                metadata, errors = extract_metadata_robustly(pdf_file, vision_client, 2)

                # If there were any errors, store them for the final report
                if errors:
                    all_metadata_errors.append({
                        "pdf_name": pdf_file.name,
                        "location": str(pdf_file.parent),
                        "errors": errors
                    })

                # If metadata extraction failed completely, we cannot proceed with this file
                if metadata is None:
                    logging.error(f"FATAL: Skipping PDF {pdf_file.name} due to critical metadata extraction failure.")
                    with map_lock:
                        pdf_to_excel_map[pdf_file.name] = "ERROR: Metadata Extraction Failed"
                    continue # Skip to the next PDF in the pair

                union_ward = metadata.get("ইউনিয়ন/ওয়ার্ড/ক্যাঃ বোঃ", "Not Found")
                voter_area = metadata.get("ভোটার এলাকার নাম", "Not Found")
                # Check if either of the values is still "Not Found"
                if union_ward == "Not Found" or voter_area == "Not Found":
                    # If so, create a specific error message
                    error_message = f"Critical field(s) missing: Union/Ward='{union_ward}', Voter Area='{voter_area}'."
                    logging.error(f"For '{pdf_file.name}': {error_message} Critical metadata NOT FOUND!")
                    
                    # Add the error details to our shared "error box" (the dictionary)
                    with map_lock:
                        # First, update the main processing map to show this file failed
                        pdf_to_excel_map[pdf_file.name] = "ERROR: Critical Metadata Missing"
                        
                        # Second, add the detailed error to our new error dictionary
                        critical_metadata_failures[pdf_file.name] = {
                            "pdf_path": str(pdf_file),
                            "reason": error_message
                        }
                    
                    


                full_text=process_one_pdf(pdf_file, bucket_name)
                if full_text:
                    #all_values = extract_all_values(full_text, keys_patterns)
                    #metadata= build_metadata(all_values)
                    all_records = find_and_parse_records(full_text,metadata)
                    if all_records:
                        write_to_excel(metadata, all_records, constituency_name, base_output_dir,pdf_file.name,pdf_to_excel_map, map_lock)
                    else:
                        #print(f"  - No voter records found in {pdf_file.name}, skipping Excel writing.")
                        logging.warning(f"No voter records found in {pdf_file.name}, skipping Excel writing.")
                
                # If everything above succeeded, mark as successful and exit the retry loop
                processed_successfully = True
                successfully_processed_pdfs.append(pdf_file) 
                #logging.info(f"  [SUCCESS]Created Excel for '{pdf_file.name}'.")
                #print(f"  [Success]Successfully processed and created Excel for '{pdf_file.name}'.")
                break # <-- Exit the for loop for attempts
            except Exception as e:
                logging.error(f"An error occurred on attempt {attempt + 1} for '{pdf_file.name}': {e}")
                #print(f"[Error]An error occurred on attempt {attempt + 1} for '{pdf_file.name}': {e}")
                if attempt < MAX_RETRIES - 1: # If this wasn't the last attempt
                    logging.info(f"Waiting for {RETRY_DELAY_SECONDS} seconds before retrying...")
                    #print(f"Waiting for {RETRY_DELAY_SECONDS} seconds before retrying...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logging.error(f"All {MAX_RETRIES} retries failed for '{pdf_file.name}'. This file will be skipped.")
                    #print(f"All {MAX_RETRIES} retries failed for '{pdf_file.name}'. This file will be skipped.")
    return successfully_processed_pdfs, all_metadata_errors
    #return f"Finished pair: {[p.name for p in pdf_pair]}"



#_______________________ Extract PDFs from a folder_________________
def extract_PDFs(root_dir):
   root_dir = Path(root_dir)
   return list(root_dir.glob("*.pdf"))



#_______________________FIND ALL LAST FOLDER FROM PDF ROOT_________________
def find_last_folders(root_dir):
    last_folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # If 'dirnames' is empty, it means there are no subdirectories
        # within the current 'dirpath', making it a "last folder".
        if not dirnames:
            last_folders.append(dirpath)
    return last_folders


#_______________________FIND INTERMEDIATE PDFS_________________
def find_intermediate_pdfs(root_dir):
    """
    Scans for PDFs located in 'parent' or 'intermediate' folders 
    (folders that have sub-directories and are ignored by find_last_folders).
    """
    intermediate_pdfs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # If 'dirnames' is present, it means this folder has subdirectories
        if dirnames:
            for f in filenames:
                if f.lower().endswith('.pdf'):
                    full_path = Path(dirpath) / f
                    intermediate_pdfs.append(full_path)
    return intermediate_pdfs



############### MAIN DRIVER FUNCTION ##############
def main():
    """
    Finds all PDF pairs from all constituencies and processes the pairs in parallel.
    """
    # --- GLOBAL CONFIGURATION ---
    #script_dir = Path('/content/drive/MyDrive')
    script_dir = Path(__file__).parent
    pdf_root   = script_dir / "pdf-voterlist"
    output_root_dir = script_dir / "Formatted_Excel_Output" # Main output folder
    log_output_dir = script_dir / "Log_Output"
    log_output_dir.mkdir(parents=True, exist_ok=True)
    service_account_filename = os.getenv("SERVICE_ACCOUNT_FILE")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(script_dir / service_account_filename)
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
    thread = os.cpu_count()

    overall_start_time = time.perf_counter()

    #Find all constituencies to process
    constituencyFolders = [folder for folder in pdf_root.glob("*") if folder.is_dir()]
    if not constituencyFolders:
        print("ERROR: No constituency folders found in 'pdf-voterlist'. Exiting.")
        return
    print(f"Found {len(constituencyFolders)} constituencies to process")
    print("="*60)

    #----------Main LOOP - PROCESS ONE CONSTITUENCY AT A TIME-------------------------
    for const_folder in constituencyFolders:
        constituency_name = const_folder.name
        pdf_to_excel_map = {}
        map_lock = threading.Lock()

        critical_metadata_failures = {}


        # --- A. SETUP FOLDERS AND LOGGING FOR THIS CONSTITUENCY ---
        constituency_output_dir = output_root_dir / constituency_name
        constituency_output_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging to go to a file *inside* the constituency's output folder
        
        log_file_path =  log_output_dir / f"{constituency_name}_processing_log.txt"


        # ---LOGGING SETUP ---
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
        # --- END OF LOGGING SETUP ---

        #print("--- Starting Parallel PDF Pair Processing ---")
        logging.info(f"--- Starting Parallel PDF Pair Processing for {constituency_name}  ---")

        # --- 2. GATHER PDF PAIRS for this constituency ---
        #constituencyFolders = [folder for folder in pdf_root.glob("*") if folder.is_dir()]
        #all_pdf_pairs = []
        all_jobs = []
        all_pdfs_in_const = []
        all_last_folders = find_last_folders(const_folder)
      
        for folder_path in all_last_folders:
            pdf_files_in_folder = extract_PDFs(folder_path)
            if pdf_files_in_folder:
                all_jobs.append({"pair": pdf_files_in_folder, "constituency": constituency_name})
                all_pdfs_in_const.extend(pdf_files_in_folder)

        if not all_jobs:
            #print("\n❌ ERROR: No PDF pairs were found for this constituency. Skipping")
            logging.info("  [ERROR]: No PDF pairs were found to process.")
            logging.info("="*60 + "\n")
            continue
        
        # Populate the dictionary with all PDF names as keys and empty values.
        for pdf_path in all_pdfs_in_const:
            pdf_to_excel_map[pdf_path.name] = ""
        #logging.info(f"Initialized map with {len(pdf_to_excel_map)} PDFs for '{constituency_name}'.")

        all_pdfs_to_process = []
        for job in all_jobs:
            all_pdfs_to_process.extend(job["pair"]) # extend adds all items from one list to another
        
        total_pdfs = len(all_pdfs_to_process)
        logging.info(f"Found a total of {len(all_jobs)} pairs ({total_pdfs} individual PDF files) to process.")
        #total_jobs = len(all_jobs)
        #print(f"A total of {len(all_jobs)} PDF pairs found to process.")
        #logging.info(f"A total of {len(all_jobs)} PDF pairs found to process.")
        total_start_time = time.perf_counter()
        all_successfully_processed_pdfs = []
        collected_metadata_errors = [] 

        
        # ---  PARALLEL PROCESSING LOOP ---
        # We will process a number of pairs at the same time.
        # max_workers=2 means we process 2 pairs (4 files) simultaneously.
        # You can increase this to 4 for your Core i3.
        
        with ThreadPoolExecutor(max_workers=thread) as executor:
            future_to_job = {
                executor.submit(process_malefemale_pdfs, job["pair"], GCS_BUCKET_NAME,job["constituency"], constituency_output_dir,pdf_to_excel_map, map_lock, critical_metadata_failures): job
                for job in all_jobs
            }

            for i, future in enumerate(as_completed(future_to_job)):
                #job = future_to_pair[future]
                #pdf_pair_names = [p.name for p in job["pair"]]
                try:
                    result_list_of_paths,  errors_from_worker  = future.result()

                    if errors_from_worker:
                        collected_metadata_errors.extend(errors_from_worker)

                    #result = future.result()
                    all_successfully_processed_pdfs.extend(result_list_of_paths)
                    completed_pairs = i + 1
                    total_pairs = len(all_jobs)
                    progress_percent = (completed_pairs / total_pairs) * 100
                    logging.info(f"Progress: {completed_pairs} of {total_pairs} pairs processed ({progress_percent:.2f}%)")
                    #completed_jobs += 1
                    #progress_percent = (completed_jobs / total_jobs) * 100
                    # The '\r' at the start moves the cursor to the beginning of the line,
                    # and 'end=""' prevents it from creating a new line.
                    # This creates a progress bar that updates in place.
                    #print(f"\rProgress: {completed_jobs} of {total_jobs} pairs processed ({progress_percent:.2f}%)", end="")
                    #logging.info(f"Progress: {completed_jobs} of {total_jobs} pairs processed ({progress_percent:.2f}%)")
                    

                    #print(f"  [COMPLETED PAIR {i+1}/{len(all_jobs)}] Success for: {pdf_pair_names}")
                except Exception as e:
                    #print(f"  [ERROR] Pair {pdf_pair_names} generated an exception: {e}")
                    logging.info(f"[ERROR] Pair {pdf_pair_names} generated an exception: {e}")

        
        #print("\n" + "="*60)
        #print("🎉 ALL PDF PAIRS PROCESSED SUCCESSFULLY!")
        #print(f"Total time taken: {(total_end_time - total_start_time) / 60:.2f} minutes")
        logging.info("="*70)
        logging.info("="*70)

        #logging.info("ALL PDF PAIRS PROCESSED SUCCESSFULLY!")
        #logging.info(f"Total time taken: {(total_end_time - total_start_time) / 60:.2f} minutes")
        logging.info("--- First run completed, looking for missing PDF files ---")

        
        set_of_all_pdfs = set(all_pdfs_to_process)
        set_of_successful_pdfs = set(all_successfully_processed_pdfs)

        missed_pdfs = list(set_of_all_pdfs - set_of_successful_pdfs)

        if not missed_pdfs:
            logging.info("--- No missing PDF found ---")

        if missed_pdfs:
            logging.info("="*60)
            logging.info(f"--- ATTEMPTING TO RE-PROCESS {len(missed_pdfs)} MISSED FILES (SEQUENTIAL RUN) ---")
            #logging.info("List of Missed PDf file(s):",missed_pdfs)
            logging.info(f"List of Missed PDf file(s): {missed_pdfs}")
            
            # This list will hold files that succeed during this final run.
            reprocessed_successes = []

            # We need to find the original "job" info for each missed PDF.
            # We'll create a quick lookup map for this.
            pdf_to_job_map = {}
            for job in all_jobs:
                for pdf_file in job["pair"]:
                    pdf_to_job_map[pdf_file] = job
            
            # Loop through each missed PDF one by one.
            for pdf_file in missed_pdfs:
                job_info = pdf_to_job_map.get(pdf_file)
                if not job_info:
                    logging.error(f"Could not find original job info for {pdf_file.name}. Cannot re-process.")
                    continue

                constituency_name = job_info["constituency"]
                
                # We call the same reliable worker function. Since it takes a list (a pair),
                # we pass the single PDF inside a list.
                logging.info(f"--Re-processing: {pdf_file.name} for constituency '{constituency_name}'")
                
                # The function returns a list of successes.
                successful_reprocesses,errors_retry = process_malefemale_pdfs(
                    [pdf_file], # Pass the single file as a list (pair of one)
                    GCS_BUCKET_NAME,
                    constituency_name,
                    output_root_dir,
                    pdf_to_excel_map,
                    map_lock,
                    critical_metadata_failures
                )
                
                # Add any successes from this run to our list.
                if successful_reprocesses:
                    reprocessed_successes.extend(successful_reprocesses)
                if errors_retry:
                    collected_metadata_errors.extend(errors_retry)

            # After the re-processing loop, update the main success list.
            all_successfully_processed_pdfs.extend(reprocessed_successes)
        logging.info("="*60)
        logging.info("--- SCANNING FOR PDFs IN INTERMEDIATE FOLDERS ---")
        
        intermediate_pdfs = find_intermediate_pdfs(const_folder)
        
        # Filter out PDFs that might have been picked up already (Safety check)
        processed_set = set(all_successfully_processed_pdfs)
        new_intermediate_pdfs = [p for p in intermediate_pdfs if p not in processed_set]

        if new_intermediate_pdfs:
            logging.info(f"Found {len(new_intermediate_pdfs)} PDF(s) in intermediate folders. Processing sequentially...")
            
            # Add these new files to our total list of files to track for the final report
            all_pdfs_to_process.extend(new_intermediate_pdfs)
            
            for pdf_file in new_intermediate_pdfs:
                logging.info(f"--Processing Intermediate File: {pdf_file.name}")
                
                # Use the existing sequential processing function
                success_inter, errors_inter = process_malefemale_pdfs(
                    [pdf_file], 
                    GCS_BUCKET_NAME, 
                    constituency_name, 
                    constituency_output_dir, 
                    pdf_to_excel_map, 
                    map_lock, 
                    critical_metadata_failures
                )
                
                if success_inter:
                    all_successfully_processed_pdfs.extend(success_inter)
                if errors_inter:
                    collected_metadata_errors.extend(errors_inter)
        else:
            logging.info("No un-processed PDFs found in intermediate folders.")



        # --- FINAL REPORT ---
        total_end_time = time.perf_counter()
        logging.info("="*70)
        logging.info("="*70)
        if missed_pdfs:
            logging.info("--- FINAL REPORT AFTER RE-PROCESSING ATTEMP ---")
        if not missed_pdfs:
            logging.info("--- FINAL SUMMARY ---")

        
        # We recalculate the missed files based on the updated success list.
        final_set_of_all_pdfs = set(all_pdfs_to_process)
        final_set_of_successful_pdfs = set(all_successfully_processed_pdfs)
        final_missed_pdfs = final_set_of_all_pdfs - final_set_of_successful_pdfs
        
        if not final_missed_pdfs:
            total_pdfs = len(all_pdfs_to_process)
            logging.info(f"SUCCESS: All {total_pdfs} PDF files were processed successfully!")
        else:
            logging.warning(f"COMPLETED WITH ISSUES: Out of TOTAL {total_pdfs} PDF file(s), {len(missed_pdfs)} PDF file(s) failed to process.")
            logging.warning("--- LIST OF UNPROCESSED FILES ---")
            # Loop through the set of missed files and log each one.
            for pdf_path in final_missed_pdfs:
                # We provide the name and the parent folder to make it easy to find.
                logging.warning(f"  - {pdf_path.name} ")
            logging.warning("-----------------------------")
        
        
        logging.info("--------------Generating Final Constituency Report ------------")
        
        # Check for duplicate Excel filenames (values)
        generated_files = [fname for fname in pdf_to_excel_map.values() if fname] # Get non-empty values
        file_counts = Counter(generated_files)
        duplicates = {filename: count for filename, count in file_counts.items() if count > 1}

        duplicate_count = 0
        if not duplicates:
            logging.info("✅ Uniqueness Check: No duplicate Excel filenames were generated.")
        else:
            logging.warning("❌ Uniqueness Check: Duplicate Excel filenames were found!")
            c= 0
            for filename, count in duplicates.items():
                c += 1
                source_pdfs = [pdf for pdf, excel in pdf_to_excel_map.items() if excel == filename]
                logging.warning(f"{c} - File '{filename}' was generated {count} times by PDFs: {source_pdfs}")
            logging.info(f"Total Number of instances of duplicate files with different names: {c}")
            duplicate_count = c

        
        logging.info("-" * 60)

        # Write the dictionary to an Excel file
        report_data = [{"Source PDF": pdf, "Generated Excel": excel} for pdf, excel in pdf_to_excel_map.items()]
        report_df = pd.DataFrame(report_data)
        report_path = log_output_dir / f"{constituency_name}_processing_report.xlsx"
        
        try:
            report_df.to_excel(report_path, index=False, engine='openpyxl')
            logging.info(f"✅ Full processing report saved to: {report_path}")
        except Exception as e:
            logging.error(f"Could not save the final report to Excel. Reason: {e}")
        
        logging.info("-" * 60)
        logging.info("--- Critical Metadata Failure Report ---")
        if not critical_metadata_failures:
            logging.info("OK: All PDFs had the required metadata for file path generation (No 'NOT FOUND' issue).")
        else:
            logging.warning(f"ATTENTION: Found {len(critical_metadata_failures)} PDF(s) with missing critical metadata with NOT FOUND Issue:")
            for pdf_name, details in critical_metadata_failures.items():
                logging.warning(f"  -> File: {pdf_name}")
                logging.warning(f"     - Path: {details['pdf_path']}")
                logging.warning(f"     - Reason: {details['reason']}")
        
        if collected_metadata_errors:
            logging.warning("--- Additional METADATA ERROR & MISMATCH REPORT ---")
            logging.warning(f"Found {len(collected_metadata_errors)} PDF(s) with metadata issues:")
            
            for report in collected_metadata_errors:
                logging.warning(f"\n  PDF File: {report['pdf_name']}")
                logging.warning(f"  Location: {report['location']}")
                for error_message in report['errors']:
                    logging.warning(f"    - Issue: {error_message}")
        logging.info("-" * 60)
        
        logger = logging.getLogger()
        verify_stats = run_verification(write_to_file=True, existing_logger=logger,const_name=const_folder,duplicate_count=duplicate_count)
        #logging.info(f"--- Finishing Constituency: {constituency_name} ---\n")

        counts_ok = verify_stats["counts_match"]
        if not critical_metadata_failures:
            metadata_ok = True
        else:
            metadata_ok = False


        log_message = f"{verify_stats['pdf_count']}/{verify_stats['excel_count']} Excels generated -- Duplicates: {verify_stats['total_duplicate_count']} -- Count Match:{counts_ok} -- Metadata OK: {metadata_ok} -- unpaired folders - PDF:{verify_stats['unpaired_pdf_folders']} Vs Excel: {verify_stats['unpaired_excel_folders']}"

        #os.makedirs("status", exist_ok=True) 
        #with open("status/logmessage.csv","a",encoding="utf-8-sig") as f:
            #f.write(f"{constituency_name},{log_message},Not uploaded\n")

        conn = get_conn()
        conn.execute(
            "INSERT INTO logmessage (constituency_name, log_message) VALUES (?, ?)",
            (constituency_name, log_message)
        )
        conn.commit()


        logging.info("="*60)
        logging.info(f"--- UPLOAD DECISION FOR {constituency_name} ---")
        logging.info(f"Criteria: Counts Match? {counts_ok} | Metadata OK? {metadata_ok}")
        logging.info(f"Log Message: {log_message}")

        if counts_ok and metadata_ok:
            logging.info("✅ PASSED All upload verification tests...")
        else:
            logging.warning("❌ FAILED. Requires manual Review...")

        '''#--- UPLOAD BASED ON VERIFICATION RESULTS ---
        if counts_ok and metadata_ok:
            logging.info("✅ PASSED. Uploading to Main Drive...")
            serviceaccount_drive_upload_module.process_and_upload_folder(constituency_name, log_message)
        
        else:
            logging.warning("❌ FAILED. Uploading for later Error/Review Drive...")
            serviceaccount_drive_upload_module.process_and_upload_folder(constituency_name, log_message)
            #error_upload_module.process_and_upload_error_folder(constituency_name, log_message)
        '''
        
        logging.info(f"Total time taken: {(total_end_time - total_start_time) / 60:.2f} minutes")
        logging.info(f"Finished Constituency: {constituency_name}-------")
        
        


    #Overall Summary
    overall_end_time = time.perf_counter()
    print("="*60)
    print(f"🎉🎉🎉 ALL {len(constituencyFolders)} CONSTITUENCIES HAVE BEEN PROCESSED. 🎉🎉🎉")
    print(f"Total script execution time: {(overall_end_time - overall_start_time) / 60:.2f} minutes.")


################# MAIN CODE #######################



# --- This makes the script runnable ---
if __name__ == "__main__":
    freeze_support()
    print("===========================================")
    print("   PHASE 1: DOWNLOADING SOURCE FILES       ")
    print("===========================================")
    downloader.main_downloader()
    print("\n")
    
    load_region_mapping()

    print("===========================================")
    print("   PHASE 2: PROCESSING & UPLOADING         ")
    print("===========================================")
    main()

   
    #print("\nPipeline completed. Now running verification and writing log for constituencies...")
    #run_verification()


