# Identify 2 outliers as seen on PCA plot and remove for further analysis

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from win32comext.mapi.mapiutil import prTable

# Load files
maldi_file = 'peak_grouping_matrix12.csv'
metadata_file = 'metadata.csv'
df = pd.read_csv(maldi_file, dtype={'SampleID': str})
metadata = pd.read_csv(metadata_file, dtype={'SampleID': str, 'Strains': str, 'bio_rep': str}) # prevent errors as it may take as number

df['SampleID'] = df['SampleID'].astype(str).str.strip()
metadata['SampleID'] = metadata['SampleID'].astype(str).str.strip()


# Initial PCA on all 35 strains
sample_ids = df.iloc[:, 0]
intensity_data = df.iloc[:, 1:].astype(float)

# Log normalization and robust scaling
log_normalized = np.log1p(intensity_data)
scaled_data = RobustScaler().fit_transform(log_normalized)

# PCA
n_components = 50

pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(scaled_data)
column_names = [f'PC{i+1}' for i in range(n_components)]
pca_df = pd.DataFrame(pca_result, columns=column_names)
pca_df.insert(0, 'SampleID', sample_ids)
merged_all = pd.merge(pca_df, metadata, on='SampleID', how='inner')

# Save PCA output with metadata
merged_all.to_csv('pca35_with_metadata.csv', index=False)

# Identify the 2 most different strains
centroid = merged_all[['PC1', 'PC2']].mean().values.reshape(1, -1)
merged_all['Distance'] = cdist(merged_all[['PC1', 'PC2']], centroid).flatten()
strain_distance = merged_all.groupby('Strains')['Distance'].mean().sort_values(ascending=False)

# Identify top 2 outlier strains
outliers = strain_distance.head(2).index.tolist()
print("Most different strains:", outliers)

# Filter out the 2 most different strains
metadata_filtered = metadata[~metadata['Strains'].isin(outliers)].copy()

sample_ids_filtered = set(metadata_filtered['SampleID'])
#df_filtered = df[df['SampleID'].isin(metadata_filtered['SampleID'])]
df_filtered = df[df['SampleID'].isin(metadata_filtered['SampleID'])].copy()
df_filtered = df_filtered.reset_index(drop=True)
sample_ids_filtered = df_filtered['SampleID']


# Re-do normalisation and scaling
sample_ids_filtered = df_filtered['SampleID']
intensity_data_filtered = df_filtered.iloc[:, 1:].astype(float)
log_normalized_filtered = np.log1p(intensity_data_filtered)
scaled_data_filtered = RobustScaler().fit_transform(log_normalized_filtered)


# PCA on filtered data
pca_filtered = PCA(n_components=n_components)
pca_result_filtered = pca_filtered.fit_transform(scaled_data_filtered)



# Merge PCA with metadata
pca_df_filtered = pd.DataFrame(pca_result_filtered, columns=column_names)
pca_df_filtered.insert(0, 'SampleID', sample_ids_filtered)
merged_filtered = pd.merge(pca_df_filtered, metadata_filtered, on='SampleID', how='inner')

# Save PCA output with metadata
merged_filtered.to_csv('pca33_with_metadata.csv', index=False)

# PCA plot of remaining 33 strains
unique_strains = merged_filtered['Strains'].unique()
palette = sns.color_palette("hsv", len(unique_strains))

plt.figure(figsize=(14, 10))

# Plot PCA with hue=Strains and style=bio_rep
scatter = sns.scatterplot(
    data=merged_filtered,
    x='PC1', y='PC2',
    hue='Strains',
    style='bio_rep',
    palette=palette,
    s=100,
    legend='full'
)

plt.title('PCA of 33 Strains (Excluding 2 Outliers)', fontsize=14)
plt.xlabel(f'PC1 ({pca_filtered.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca_filtered.explained_variance_ratio_[1]*100:.2f}%)')

# Extract strain labels and handles for legend
strain_labels = [str(s) for s in merged_filtered['Strains'].unique()]
bio_rep_labels = [str(b) for b in merged_filtered['bio_rep'].unique()]

handles, labels = scatter.get_legend_handles_labels()

# Seaborn default marker cycle for style
default_markers = ['o', 'X', 's', 'P', 'D', '^']

# Map bio_rep categories to markers
bio_rep_marker_map = {bio_rep: default_markers[i] for i, bio_rep in enumerate(bio_rep_labels)}

# Strain handles from seaborn legend
strain_handles = handles[1:1+len(strain_labels)]

# Create custom handles for bio_rep legend
custom_bio_rep_handles = [
    mlines.Line2D([], [], color='black', marker=bio_rep_marker_map[bio_rep], linestyle='None', markersize=10)
    for bio_rep in bio_rep_labels
]

# Draw strain legend
strain_legend = plt.legend(
    handles=strain_handles,
    labels=strain_labels,
    title='Strains',
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    borderaxespad=0.,
    ncol=2,
    fontsize='small',
    title_fontsize='medium'
)

plt.gca().add_artist(strain_legend)  # keep strain legend visible

# Draw biological replicate legend
plt.legend(
    handles=custom_bio_rep_handles,
    labels=bio_rep_labels,
    title='Biological Replicates',
    bbox_to_anchor=(1.02, 0.45),
    loc='upper left',
    borderaxespad=0.,
    fontsize='small',
    title_fontsize='medium'
)

plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig('pca_33_strains.png', dpi=300)
plt.show()



