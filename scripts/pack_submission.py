"""
Automated Packager for Final Hackathon Submission
Amazon ML Challenge 2026
"""

import os
import sys
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
CODE_DIR = BASE_DIR / "code"
DOC_PATH = BASE_DIR / "Documentation_template.md"
ZIP_PATH = BASE_DIR / "Data_Titans_submission.zip"

def create_submission_zip():
    print(f"Creating submission package: {ZIP_PATH.name}...")
    
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. Add output files
        matching_file = OUTPUT_DIR / "matching_results.tsv"
        candidate_file = OUTPUT_DIR / "candidate_pairs.tsv"
        
        if matching_file.exists():
            print(f"Adding output/matching_results.tsv ({matching_file.stat().st_size / (1024*1024):.1f} MB)...")
            z.write(matching_file, "output/matching_results.tsv")
        else:
            print("ERROR: matching_results.tsv does not exist yet!")
            return False
            
        if candidate_file.exists():
            print(f"Adding output/candidate_pairs.tsv ({candidate_file.stat().st_size / (1024*1024):.1f} MB)...")
            z.write(candidate_file, "output/candidate_pairs.tsv")
        else:
            print("ERROR: candidate_pairs.tsv does not exist yet!")
            return False
            
        # 2. Add code directory
        for root, dirs, files in os.walk(CODE_DIR):
            if "__pycache__" in root:
                continue
            for file in files:
                if file.endswith(".pyc"):
                    continue
                full_path = Path(root) / file
                rel_path = full_path.relative_to(BASE_DIR)
                print(f"Adding {rel_path}...")
                z.write(full_path, str(rel_path).replace("\\", "/"))
                
        # 3. Add Documentation_template.md
        if DOC_PATH.exists():
            print(f"Adding Documentation_template.md...")
            z.write(DOC_PATH, "Documentation_template.md")
        else:
            print("ERROR: Documentation_template.md does not exist!")
            return False

    print(f"\nSuccessfully packaged submission zip: {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024*1024):.1f} MB)!")
    return True

if __name__ == "__main__":
    create_submission_zip()
