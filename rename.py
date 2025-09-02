# rename files to be easier to read from codes

import os

# Paths to 4 folders
folder_paths = [
    "D://Research project//MALDI data//Protein_trimmed//1bio",
    "D://Research project//MALDI data//Protein_trimmed//2bio",
    "D://Research project//MALDI data//Protein_trimmed//3bio",
    "D://Research project//MALDI data//Protein_trimmed//4bio"
]

# sort folder paths
all_files = []

# Collect all file paths
for folder in folder_paths:
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            full_path = os.path.join(folder, filename)
            all_files.append(full_path)

# Sort the list of files
all_files.sort()


# Rename files
for i, file_path in enumerate(all_files, start=1):
    folder = os.path.dirname(file_path)
    new_name = f"{i}.txt"
    new_path = os.path.join(folder, new_name)

    # Avoid overwriting existing numbered files
    if os.path.exists(new_path):
        print(f"File {new_path} already exists. Skipping.")
        continue

    os.rename(file_path, new_path)

