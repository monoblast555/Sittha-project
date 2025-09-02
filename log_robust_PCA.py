# Normalise and scale by the chosen methods, log and robust. And plot PCA to see result clearly

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
maldi_file = 'peak_grouping_matrix12.csv'
df = pd.read_csv(maldi_file)

sample_ids = df.iloc[:, 0]
intensity_data = df.iloc[:, 1:].astype(float)

# Load metadata
metadata_file = 'metadata.csv'
metadata = pd.read_csv(metadata_file)

# Preprocessing: log & robust
log_normalized = np.log1p(intensity_data)
scaler = RobustScaler()
scaled_data = scaler.fit_transform(log_normalized)

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Create PCA dataframe
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df.insert(0, 'SampleID', sample_ids)

# Merge PCA with metadata
merged_df = pd.merge(pca_df, metadata, on='SampleID', how='inner')

# Save PCA output with metadata
merged_df.to_csv('pca_with_metadata.csv', index=False)

# Create a color palette
unique_strains = merged_df['Strains'].unique()
palette = sns.color_palette("hsv", len(unique_strains))

# Plot PCA
import matplotlib.lines as mlines

plt.figure(figsize=(14, 10))

# PCA scatterplot
scatter = sns.scatterplot(
    data=merged_df,
    x='PC1', y='PC2',
    hue='Strains',
    style='bio_rep',
    palette=palette,
    s=100,
    legend='full'
)

plt.title('PCA of MALDI-TOF Data (Colored by Strains)', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')

# Extract handles & labels from Seaborn plot
handles, labels = scatter.get_legend_handles_labels()

# Get strain and bio_rep values
strain_labels = [str(s) for s in merged_df['Strains'].unique()]
bio_rep_labels = [str(b) for b in merged_df['bio_rep'].unique()]

# Seaborn default marker cycle
default_markers = ['o', 'X', 's', 'P', 'D', '^']

# Map bio_rep to marker
bio_rep_marker_map = {
    rep: default_markers[i]
    for i, rep in enumerate(bio_rep_labels)
}

# Strain handles
strain_handles = handles[1:1+len(strain_labels)]

# Custom bio_rep handles with visible markers
custom_bio_rep_handles = [
    mlines.Line2D([], [], color='black', marker=bio_rep_marker_map[rep],
                  linestyle='None', markersize=10)
    for rep in bio_rep_labels
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

plt.gca().add_artist(strain_legend)  # keep strain legend

# Draw bio_rep legend
plt.legend(
    handles=custom_bio_rep_handles,
    labels=bio_rep_labels,
    title='Biological Replicates',
    bbox_to_anchor=(1.02, 0.5),
    loc='upper left',
    borderaxespad=0.,
    fontsize='small',
    title_fontsize='medium'
)

plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave space on the right for legends
plt.savefig('pca_with_35_strains.png', dpi=300)
plt.show()
