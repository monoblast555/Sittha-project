import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Load smoothed data
df = pd.read_csv("smoothed_protein.csv")

# Separate sample IDs and spectra
sample_ids = df.iloc[:, 0]
spectra = df.iloc[:, 1:].to_numpy()
mz = df.columns[1:].astype(float)


# Store detected peaks info for all samples
all_peak_indices = []
all_peak_mz = []

for i, spectrum in enumerate(spectra):
    peaks, props = find_peaks(spectrum,
        height= 400,
        threshold= None,
        prominence= 800,
        width= 50
    )
    all_peak_indices.append(peaks)
    all_peak_mz.append(mz[peaks])

    # Plot first 12 samples with detected peaks to see from 3 different strains
    if i < 12:
        plt.figure(figsize=(10,4))
        plt.plot(mz, spectrum, label='Smoothed Spectrum')
        plt.plot(mz[peaks], spectrum[peaks], 'x', color='red', label='Detected Peaks')
        plt.title(f'Sample ID: {sample_ids[i]} - Peak Detection')
        plt.xlabel('m/z')
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()

# Create a unified list of all detected peak m/z across all samples
unique_peaks = np.unique(np.concatenate(all_peak_mz))

# Build peak intensity matrix: rows = samples, cols = unique peaks
peak_matrix = np.zeros((spectra.shape[0], len(unique_peaks)))

for i, spectrum in enumerate(spectra):
    peaks = all_peak_indices[i]
    peak_positions = mz[peaks]
    peak_intensities = spectrum[peaks]

    # Assign intensities to closest unique peak column
    for pos, intensity in zip(peak_positions, peak_intensities):
        idx = np.argmin(np.abs(unique_peaks - pos))
        peak_matrix[i, idx] = intensity

# Save peak intensity matrix to CSV
peak_df = pd.DataFrame(peak_matrix, columns=[f"{p:.4f}" for p in unique_peaks])
peak_df.insert(0, 'SampleID', sample_ids)
peak_df.to_csv('peak_matrix.csv', index=False)



