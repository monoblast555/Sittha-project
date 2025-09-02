# Spectra alignment

import pandas as pd
import numpy as np
import icoshift

# Load the filtered CSV
df = pd.read_csv("protein_1000_10000.csv")

# Extract Sample IDs
sample_ids = df.iloc[:, 0].values

# Extract intensity matrix (samples x m/z)
intensity_matrix = df.iloc[:, 1:].to_numpy()

# Check data
print("Target check:", type(intensity_matrix), intensity_matrix.shape)
print("Any NaNs:", np.isnan(intensity_matrix).any())
print("Any all-zero rows:", np.any(intensity_matrix.sum(axis=1) == 0))

# Filter out bad rows to prevent errors
valid_rows = ~np.isnan(intensity_matrix).any(axis=1) & (intensity_matrix.sum(axis=1) != 0)
intensity_matrix = intensity_matrix[valid_rows].astype(np.float64)
sample_ids = sample_ids[valid_rows]

# Check shape before alignment
print(f"Shape before alignment: {intensity_matrix.shape}")

# Compute the target spectrum as the mean of the intensity matrix
target_spectrum = np.mean(intensity_matrix, axis=0).reshape(1, -1)

print("target_spectrum shape:", target_spectrum.shape)
print("intensity_matrix shape:", intensity_matrix.shape)

# Align using explicit target
xCS, ints, ind, _ = icoshift.icoshift(target_spectrum, intensity_matrix)

# Convert back to dataframe
aligned_df = pd.DataFrame(xCS, columns=df.columns[1:])
aligned_df.insert(0, 'SampleID', sample_ids)

# Save file
aligned_df.to_csv("aligned_protein.csv", index=False)

