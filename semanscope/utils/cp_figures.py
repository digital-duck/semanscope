import os

# The target directory for the copy operation
TARGET_DIR = "/home/papagame/projects/Proj-ZiNets/zinets/docs/conference/acl/latex/geo-meaning/figures/ACL"
BASH_SCRIPT_NAME = "cp_figures.sh"


text = """
list of Figures

# Datasets

## Alphabets

- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/alphabets-phate-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/alphabets-phate-qwen3-06b-enu-kor.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/alphabets-phate-qwen3-06b-chn-kor-jpn.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/echarts/echarts-acl-1-alphabets-qwen3-embedding-0.6b-phate-chn-kor-jpn-deu-enu-ara.png

### radicals-kangxi

- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/radicals-kangxi-phate-qwen3-06b-chn.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/zinets-phate-qwen3-06b-chn.pdf


## word-v2

- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-qwen3-06b-chn.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-qwen3-06b-deu.pdf

## network word

- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/network-work-light-v2-phate-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/acl-3-network-child-v2-phate-qwen3-06b-chn.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/network-haus-arbeit-v2-phate-qwen3-06b-deu.pdf

## PeterG word
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/peterg-all-phate-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/peterg-adj-phate-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/peterg-noun-phate-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/peterg-verb-phate-qwen3-06b-enu.pdf

## Sentences

- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/li-bai-moonlight-phate-qwen3-4b-chn.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/li-bai-moonlight-phate-sentence-bert-chn.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/li-bai-moonlight-phate-openai-ada-002-chn.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/li-bai-moonlight-phate-gemini-001-chn.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/li-bai-waterfall-phate-qwen3-4b-chn.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/frost-road-phate-qwen3-4b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/frost-woods-phate-qwen3-4b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/du-fu-climb-phate-qwen3-4b-chn.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/du-fu-rain-phate-qwen3-4b-chn.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/wordsworth-strange-phate-qwen3-4b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/wang-wei-lodge-phate-qwen3-4b-chn.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/wang-wei-sickness-phate-qwen3-4b-chn.pdf


## number

- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/numbers-phate-qwen3-06b-enu.pdf


## Emoji
qwen3-06b not good
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/emoji-phate-sentence-bert-enu.pdf

- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/emoji-phate-qwen3-4b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/emoji-phate-qwen3-4b-chn-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/emoji-phate-qwen3-06b-chn-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/emoji-phate-qwen3-8b-chn-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/emoji-phate-gemini-001-chn-enu.pdf

openai-small model not good
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/emoji-phate-openai-3-small-chn-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/emoji-phate-openai-3-large-chn-enu.pdf


# Methods

- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-trimap-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-pacmap-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-umap-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-isomap-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-lle-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-spectral-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-pca-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-k-pca-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-mds-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-forceatlas2-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-t-sne-qwen3-06b-enu.pdf

# Models

- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-qwen3-06b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-qwen3-4b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-qwen3-8b-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-sentence-bert-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-mbert-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-labse-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-xlm-r-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-u-encoder-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-gemma-300m-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-gemini-001-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-openai-3-large-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-openai-3-small-enu.pdf
- /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/images/PDF/word-v2-phate-openai-ada-002-enu.pdf


"""


# 1. Extract file paths
file_paths = []
for line in text.splitlines():
    # Check if the line starts with "- " and contains the expected path structure
    if line.strip().startswith("-") and "/home/papagame/projects/" in line:
        # Extract the path (remove the leading "- ")
        path = line.split("-", 1)[1].strip()
        if path:
            file_paths.append(path)

# 2. Generate the Bash script content
bash_script_content = f"""#!/bin/bash

# --- Start of Generated Copy Script ---
# This script was generated by a Python script to copy figure files.
# It uses 'cp -f' to **force overwrite** existing files in the target directory.

# Define the target directory
TARGET_DIR="{TARGET_DIR}"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Check if the directory was created successfully (or already exists)
if [ $? -ne 0 ]; then
    echo "Error: Could not create or access the target directory: $TARGET_DIR"
    exit 1
fi

echo "Copying {len(file_paths)} files to '$TARGET_DIR' (using cp -f for overwrite)..."

# Array of source files to copy
SOURCE_FILES=(
"""
# Add all extracted file paths to the Bash array
for path in file_paths:
    # Use double quotes around the path
    bash_script_content += f'  "{path}"\n'

bash_script_content += """)

# Loop through the array and copy each file
for file_path in "${SOURCE_FILES[@]}"; do
    if [ -f "$file_path" ]; then
        # *** CHANGED: Using 'cp -f' to force overwrite without prompt ***
        cp -f "$file_path" "$TARGET_DIR/"
        # Uncomment the line below for verbose output:
        # echo "Copied: $(basename "$file_path")"
    else
        echo "Warning: File not found, skipping: $file_path"
    fi
done

echo "Copy operation complete."
# --- End of Generated Copy Script ---
"""

# 3. Write the Bash script to a file
with open(BASH_SCRIPT_NAME, "w") as f:
    f.write(bash_script_content)

print(f"✅ Extracted {len(file_paths)} file paths.")
print(f"✅ Generated Bash script: **{BASH_SCRIPT_NAME}**")
print(f"Target directory for files: **{TARGET_DIR}**")