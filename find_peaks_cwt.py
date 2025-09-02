import pandas as pd
import numpy as np
from scipy.signal import find_peaks_cwt
import matplotlib.pyplot as plt

# Load smoothed data
df = pd.read_csv("smoothed_protein.csv")

sample_ids = df.iloc[:, 0]
spectra = df.iloc[:, 1:].to_numpy()
mz = df.columns[1:].astype(float)

# Parameters for peak detection
widths = np.arange(100, 400)
min_intensity = 200          # Filter weak peaks, min = 200

# Store all detected peaks per sample
peak_indices_all = []
peak_mz_all = []

for i, spectrum in enumerate(spectra):
    peaks = find_peaks_cwt(spectrum, widths)
    filtered_peaks = [p for p in peaks if spectrum[p] >= min_intensity]
    peak_indices_all.append(filtered_peaks)
    peak_mz_all.append(mz[filtered_peaks])

    # Plot the first 12 samples to see 3 different strains
    if i < 12:
        plt.figure(figsize=(10, 4))
        plt.plot(mz, spectrum, label='Smoothed Spectrum', alpha=0.7)
        plt.plot(mz[filtered_peaks], spectrum[filtered_peaks], 'x', color='red', label='Detected Peaks')
        plt.title(f'Sample ID: {sample_ids[i]} - Peak CWT')
        plt.xlabel('m/z')
        plt.ylabel('Intensity')
        plt.legend()
        plt.tight_layout()
        plt.show()



# Build full list of unique m/z values (rounded to 4 decimals to avoid float duplicates)
all_peak_mz = np.unique(np.round(np.concatenate(peak_mz_all), 4))

# Initialize matrix
peak_matrix = np.zeros((spectra.shape[0], len(all_peak_mz)))

# Fill matrix
for i, (indices, mz_values) in enumerate(zip(peak_indices_all, peak_mz_all)):
    for idx in indices:
        rounded_mz = round(mz[idx], 4)
        col_idx = np.where(all_peak_mz == rounded_mz)[0][0]
        peak_matrix[i, col_idx] = spectra[i, idx]

# Create DataFrame and save
peak_df = pd.DataFrame(peak_matrix, columns=[f"{m:.4f}" for m in all_peak_mz])
peak_df.insert(0, 'SampleID', sample_ids)
peak_df.to_csv("peak_matrix_cwt.csv", index=False)
