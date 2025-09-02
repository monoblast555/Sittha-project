# Smooth the curves

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("corrected_protein.csv")


# Separate sample IDs and m/z values
sample_ids = df.iloc[:, 0]
spectra = df.iloc[:, 1:].to_numpy()

# smoothing to each spectrum
smoothed_spectra = np.array([
    savgol_filter(row, window_length=101, polyorder=3, deriv=0, delta=1.0, mode='interp')
    for row in spectra
])

# Visualize before and after to adjust parameters
i = 50  # Sample index
mz = df.columns[1:].astype(float)

plt.plot(mz, spectra[i], label="Original", alpha=0.6)
plt.plot(mz, smoothed_spectra[i], label="Smoothed", linewidth=2)
plt.title(f"Sample ID: {sample_ids[i]}")
plt.xlabel("m/z")
plt.ylabel("Intensity")
plt.legend()
plt.show()

# convert to dataframe and save to csv
smoothed_df = pd.DataFrame(smoothed_spectra, columns=df.columns[1:])
smoothed_df.insert(0, 'SampleID', sample_ids)
smoothed_df.to_csv("smoothed_protein.csv", index=False)

