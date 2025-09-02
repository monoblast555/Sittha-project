# Trim m/z again as there are too many blanks

import pandas as pd

# file loading
df = pd.read_csv('protein.csv')

# Separate the Sample ID column
sample_ids = df.iloc[:, 0]

# Extract and convert the m/z columns (everything except the first column)
mz_columns = pd.to_numeric(df.columns[1:], errors='coerce')
mz_data = df.iloc[:, 1:]

# Create a boolean mask for m/z <= 10000
mask = mz_columns <= 10000

# Apply the mask to filter columns
filtered_mz_data = mz_data.loc[:, mask]

# Combine Sample ID back with filtered m/z data
filtered_df = pd.concat([sample_ids, filtered_mz_data], axis=1)

# Save to a new CSV
filtered_df.to_csv('protein_1000_10000.csv', index=False)