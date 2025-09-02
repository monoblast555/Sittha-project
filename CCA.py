import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import cm

# Load PCA scores with metadata
data = pd.read_csv('pca33_with_metadata.csv')

# Prepare X matrix
# Select first 10 PCs for input to CCA
pca_columns = [col for col in data.columns if col.startswith('PC')]
X = data[pca_columns[:10]].values

# Prepare Y matrix (binary-coded labels)
strain_labels = data[['Strains']]
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(strain_labels)

# CCA
cca = CCA(n_components=3)  # Get first 3 canonical variates
X_c, Y_c = cca.fit_transform(X, Y)

# Add canonical variates to DataFrame for plotting
data['CV1'] = X_c[:, 0]
data['CV2'] = X_c[:, 1]
data['CV3'] = X_c[:, 2]

# Plot CV

strain_labels = data['Strains'].unique()
num_classes = len(strain_labels)

# Use the same consistent colormap
colors = cm._colormaps['nipy_spectral'].resampled(num_classes)
palette = {strain: mcolors.rgb2hex(colors(i)) for i, strain in enumerate(strain_labels)}

# Plot Canonical Variates
plt.figure(figsize=(12, 9))
sns.scatterplot(
    data=data,
    x='CV1', y='CV2',
    hue='Strains',
    palette=palette,
    s=80,
    edgecolor='black',
    alpha=0.85
)
plt.title('Canonical Variates from CCA (CV1 vs CV2)', fontsize=14)
plt.xlabel('Canonical Variate 1')
plt.ylabel('Canonical Variate 2')
plt.legend(
    title='Strains',
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    ncol=2,
    fontsize='small',
    title_fontsize='medium'
)
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig('cca_cv1_vs_cv2_33_strains.png', dpi=300)
plt.show()


# Convert to DataFrames for better readability
X_df = pd.DataFrame(X, columns=pca_columns[:10])
Y_df = pd.DataFrame(Y, columns=encoder.get_feature_names_out(['Strains']))


# Save to CSV just to see what's in, to show Dr Yun if they are correct
X_df.to_csv('X_matrix_PCA_scores.csv', index=False)
Y_df.to_csv('Y_matrix_onehot_labels.csv', index=False)


from scipy.spatial.distance import pdist, squareform

# Group by strain and compute centroids
centroids = data.groupby('Strains')[['CV1', 'CV2']].mean()

# Compute pairwise Euclidean distances between centroids
dist_matrix = pd.DataFrame(
    squareform(pdist(centroids.values, metric='euclidean')),
    index=centroids.index,
    columns=centroids.index
)

# Compute mean distance of each strain to all others
mean_distances = dist_matrix.mean(axis=1).sort_values(ascending=False)

# Show top 5 most isolated strains
print("Strains farthest from the rest (mean centroid distance):")
print(mean_distances.head(5))


# Visualise centroid and distance
plt.figure(figsize=(12, 9))
sns.scatterplot(data=data, x='CV1', y='CV2', hue='Strains', palette=palette, s=60, alpha=0.6)

# Plot centroids
plt.scatter(centroids['CV1'], centroids['CV2'], c='black', s=100, marker='X', label='Centroids')

for strain, row in centroids.iterrows():
    plt.text(row['CV1'], row['CV2'], strain, fontsize=8, weight='bold', ha='center', va='center')

plt.title('Canonical Variates with Strain Centroids')
plt.xlabel('CV1')
plt.ylabel('CV2')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize='small')
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.show()
