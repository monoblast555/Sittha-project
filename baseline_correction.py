from airPLS import airPLS
import pandas as pd

# load the aligned protein data
df = pd.read_csv("aligned_protein.csv")

# separate the sample column for new dataframe
samples = df.iloc[:, 0]

# m/z data columns for baseline correction
mz_data = df.iloc[:, 1:]

# baseline calculation for every rows
baseline_prot = [] # empty list for baseline data

for i in range(len(mz_data)):
    spectrum = mz_data.iloc[i].values.astype(float)  # ensure it's a float array to prevent error
    baseline = airPLS(spectrum, lambda_=1e6, porder=1, itermax=15)
    baseline_prot.append(baseline)

# convert corrected data list to new dataframe
baseline_df = pd.DataFrame(baseline_prot, columns=mz_data.columns)
baseline_df.insert(0, 'sample', samples)
baseline_df.to_csv("baseline_protein.csv", index=False)

# subtract baseline from df
corrected_values = mz_data.subtract(baseline_df.iloc[:, 1:])
corrected_df = pd.concat([samples, corrected_values], axis=1)

# save to CSV
corrected_df.to_csv("corrected_protein.csv", index=False)
