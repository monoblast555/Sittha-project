import os
import pandas as pd

folder_path = "D://Research project//MALDI data//Protein_trimmed//all"
all_data = {}

# Read each file
for filename in sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0])):
    if filename.endswith(".txt"):
        sample_name = os.path.splitext(filename)[0]
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Skip first line
            mz_intensity = [line.strip().split('\t') for line in lines if line.strip()]
            mz_dict = {float(mz): float(intensity) for mz, intensity in mz_intensity}
            all_data[sample_name] = mz_dict

# Create DataFrame
df = pd.DataFrame.from_dict(all_data, orient='index')
df.index.name = 'Sample'
df = df.sort_index(axis=1) # sort m/z columns

# Fill missing values with NA, if there's one to prevent errors
df = df.fillna('NA')

# Save to CSV
df.to_csv("protein.csv")
