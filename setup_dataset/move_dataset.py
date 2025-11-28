# move_data.py

import os
import shutil
from pathlib import Path


# Update this path to wherever the dataset is downloaded on your system
# After you run the download_UNSW_NB15.py script, check the printed path
# Also, don't push it
SOURCE_DIR_STR = r""

# The destination path is the 'data' folder in current project directory
DEST_DIR = Path.cwd() / "data"

REQUIRED_FILES = [
    "UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv"
]

if not DEST_DIR.exists():
    DEST_DIR.mkdir()
    print(f"Created destination directory: {DEST_DIR}")

SOURCE_DIR = Path(SOURCE_DIR_STR)

if not SOURCE_DIR.exists():
    print(f"Source directory not found: {SOURCE_DIR}")
else:
    print(f"\nMoving files from: {SOURCE_DIR}")
    
    # Move the required files
    for file_name in REQUIRED_FILES:
        source_file_path = SOURCE_DIR / file_name
        dest_file_path = DEST_DIR / file_name
        
        if source_file_path.exists():
            shutil.move(str(source_file_path), str(dest_file_path))
            print(f"Moved: {file_name}")
        else:
            print(f"File not found in source: {file_name}")

    # Clean up the empty cache directory structure after moving files
    try:
        # Try to remove the version folder and its parents if they are empty
        SOURCE_DIR.rmdir() 
        print(f"\nCleaned up empty directory: {SOURCE_DIR.name}")
    except OSError:
        # Directory was not empty (e.g., has other files/folders)
        print(f"\nSource directory {SOURCE_DIR.name} was not empty, skipping cleanup.")

print("\n---")
print("File moving complete")