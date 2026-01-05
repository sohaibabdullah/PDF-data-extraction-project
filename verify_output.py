# verify_output.py

import logging
import sys
from pathlib import Path
from collections import Counter
import os

def find_last_subfolders(root_path):
    """Walks a directory tree and finds all folders that have no subdirectories."""
    last_folders = []
    if not root_path.is_dir():
        return last_folders
        
    for dirpath, dirnames, _ in os.walk(root_path):
        if not dirnames:
            last_folders.append(Path(dirpath))
    return last_folders

def run_verification(write_to_file=True,existing_logger=None,const_name=None,duplicate_count=0):
    """
    Performs a detailed verification audit, including duplicate and pairing checks.
    """
    Total_duplicate_count = duplicate_count
    if existing_logger:
      logger = existing_logger
    else:
        script_dir = Path(__file__).parent
        log_output_dir = script_dir / "Log_Output"

        
        # --- 1. SETUP LOGGER (UNCHANGED) ---
        logger = logging.getLogger("verification_logger")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if logger.hasHandlers():
            logger.handlers.clear()

        # --- 2. CONFIGURE HANDLERS (UNCHANGED) ---
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_formatter = logging.Formatter('%(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
        if write_to_file:
            log_output_dir.mkdir(parents=True, exist_ok=True)
            verification_log_path = log_output_dir / "Final_verification_report.log"
            file_handler = logging.FileHandler(verification_log_path, mode='w', encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    # --- 3. RUN THE VERIFICATION LOGIC ---
    script_dir = Path(__file__).parent
    pdf_root_dir = script_dir / "pdf-voterlist"
    excel_root_dir = script_dir / "Formatted_Excel_Output"
    logger.info("x"*70)
    logger.info("="*70)
    logger.info("x"*70)
    logger.info("="*70)
    logger.info("--- INITIATING VERIFICATION AUDIT ---")
    logger.info(f"Scanning PDF Source: {pdf_root_dir}")
    logger.info(f"Scanning Excel Output: {excel_root_dir}")
    logger.info("="*70)

    if not (pdf_root_dir.is_dir() and excel_root_dir.is_dir()):
        logger.error("ERROR: Source or Output directory not found. Aborting verification.")
        return
    
    if const_name!=None:
        constituency_folders = [const_name]
        #constituency_folders = constituency_folders.append(const_name)   
    else:
        constituency_folders = [d for d in pdf_root_dir.iterdir() if d.is_dir()]
    total_discrepancies = 0
    stats = {
        "pdf_count": 0,
        "excel_count": 0,
        "unpaired_excel_folders": 0,
        "counts_match": False
        "total_duplicate_count": 0
    }
    for const_folder in constituency_folders:
        constituency_name = const_folder.name
        output_const_folder = excel_root_dir / constituency_name
        
        logger.info(f"Auditing Constituency: '{constituency_name}'...")
        logger.info("=" * 40)

        # --- CHECK 1: DUPLICATE PDF FILENAMES ---
        logger.info("  CHECK 1: Searching for duplicate PDF filenames...")
        all_pdf_files = list(const_folder.rglob("*.pdf"))
        all_pdf_names = [p.name for p in all_pdf_files]
        name_counts = Counter(all_pdf_names)
        duplicates = {name: count for name, count in name_counts.items() if count > 1}
        if not duplicates:
            logger.info("    -> OK: No duplicate PDF filenames found.")
        else:
            logger.warning("    -> WARNING: Duplicate PDF filenames found!")
            pdf_dupli_count=0
            for name, count in duplicates.items():
                pdf_dupli_count += 1
                logger.warning(f"== {pdf_dupli_count} - Filename '{name}' appears {count} times.")
            
            logger.warning(f"Total duplicate PDFs: {pdf_dupli_count}")
            Total_duplicate_count += pdf_dupli_count
            stats["total_duplicate_count"] = Total_duplicate_count

        logger.info("="*70)

        # --- CHECK 2: PDF and EXCEL FILE PAIRING ---
        logger.info("\n  CHECK 2: Verifying file pairs in last subdirectories...")
        # Check PDF pairs
        pdf_last_folders = find_last_subfolders(const_folder)
        unpaired_pdfs_found = False
        unpaired_pdfs_count = 0
        for folder in pdf_last_folders:
            pdf_count = len(list(folder.glob("*.pdf")))
            if pdf_count == 1:
                unpaired_pdfs_count += 1
                logger.warning(f"== {unpaired_pdfs_count} -> WARNING (PDFs): Found a single, unpaired PDF file in folder: {folder}")
                unpaired_pdfs_found = True
        
        if unpaired_pdfs_found:
            logger.warning(f"Total unpaired pdfs found: {unpaired_pdfs_count}")

        if not unpaired_pdfs_found:
            logger.info("    -> OK: All source PDF folders have file pairs.")
        logger.info("-"*60)
        
        # Check Excel pairs
        excel_last_folders = find_last_subfolders(output_const_folder)
        unpaired_excels_found = False
        unpaired_excels_count = 0
        if not output_const_folder.is_dir():
            logger.info("    -> INFO: Output folder for this constituency doesn't exist yet, skipping Excel pair check.")
        else:

            for folder in excel_last_folders:
                excel_count = len(list(folder.glob("*.xlsx")))
                if excel_count == 1:
                    unpaired_excels_count += 1
                    logger.warning(f"== {unpaired_excels_count}-> WARNING (Excels): Found a single, unpaired Excel file in folder: {folder}")
                    unpaired_excels_found = True
            if unpaired_excels_found:
                logger.info(f"Total upaired excels found: {unpaired_excels_count}")
            if not unpaired_excels_found:
                logger.info("    -> OK: All output Excel folders have file pairs.")
            stats["unpaired_excel_folders"] = unpaired_excels_count

        logger.info("="*70)
        # --- CHECK 3: TOTAL FILE COUNT (Summary) ---
        logger.info("\n  CHECK 3: Comparing total file counts...")
        pdf_count = len(all_pdf_files)
        excel_count = len(list(output_const_folder.rglob("*.xlsx"))) if output_const_folder.is_dir() else 0
        
        logger.info(f"    - Total Source PDFs found: {pdf_count}")
        logger.info(f"    - Total Generated Excels found: {excel_count}")
        stats["pdf_count"] = pdf_count
        stats["excel_count"] = excel_count
        if pdf_count == excel_count:
            logger.info("    -> OK: Total counts are matching.\n")
            stats["counts_match"] = True
        else:
            if Total_duplicate_count >0:
                logger.warning(f"    -> MISMATCH: Discrepancy of {abs(pdf_count - excel_count)} file(s). Note: There are {Total_duplicate_count} duplicate PDF filenames which may affect this count.\n")
                if abs(pdf_count - excel_count) == Total_duplicate_count:
                    logger.warning("       The count discrepancy matches the number of duplicate PDF filenames.")
                    stats["counts_match"] = True
            else:
                logger.warning(f"    -> MISMATCH: Discrepancy of {abs(pdf_count - excel_count)} file(s).\n")
                total_discrepancies += 1

    # --- FINAL SUMMARY ---
    logger.info("="*70)
    logger.info("--- VERIFICATION AUDIT SUMMARY ---")
    if total_discrepancies == 0:
        logger.info("SUCCESS: Passed the total file count check.")
    else:
        logger.warning(f"ATTENTION: Found count discrepancies in {total_discrepancies} constituency/constituencies.")
    logger.info("="*70)
    return stats


# --- This makes the script runnable directly from the command prompt ---
if __name__ == "__main__":
    print("--- Running Verification in Standalone (Console-Only) Mode ---")
    run_verification(write_to_file=False)