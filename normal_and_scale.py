# Try to pair normalisation and scaling methods and plot PCA scores in grid to see which is the best combination

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, Normalizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and metadata
data_df = pd.read_csv("peak_grouping_matrix12.csv")
meta_df = pd.read_csv("metadata.csv")


# First column is Sample ID
sample_ids = data_df.iloc[:, 0]
mz_values = data_df.columns[1:].astype(float)
intensity_data = data_df.iloc[:, 1:].values

hue_col = "Strains"
style_col = "bio_rep"

# Merge metadata with sample IDs
meta_df = meta_df.set_index("SampleID").loc[sample_ids].reset_index()

# Normalization functions
def normalize_none(data):  # No normalization
    return data

def normalize_tic(data):
    return data / data.sum(axis=1, keepdims=True)

def normalize_max(data):
    return data / data.max(axis=1, keepdims=True)

def normalize_log(data):
    return np.log1p(data)

# Scaling functions
def scale_none(data):  # No scaling
    return data

def scale_zscore(data):
    return StandardScaler().fit_transform(data)

def scale_minmax(data):
    return MinMaxScaler().fit_transform(data)

def scale_log(data):
    return np.log1p(data)  # log(1 + x), handles zeros

def scale_pareto(data):
    std_sqrt = np.sqrt(np.std(data, axis=0, ddof=1))
    std_sqrt[std_sqrt == 0] = 1  # avoid division by zero
    return data / std_sqrt

def scale_unit_vector(data):
    return Normalizer(norm='l2').fit_transform(data)

def scale_maxabs(data):
    return MaxAbsScaler().fit_transform(data)

def scale_robust(data):
    return RobustScaler().fit_transform(data)


# PCA functions
def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    explained = pca.explained_variance_ratio_ * 100
    return principal_components, explained

def plot_pca_with_metadata(pca_data, explained, method_name, metadata, hue="Strains", style="bio_rep"):
    plt.figure(figsize=(7, 5))
    df_plot = pd.DataFrame({
        "PC1": pca_data[:, 0],
        "PC2": pca_data[:, 1],
        hue: metadata[hue],
        style: metadata[style]
    })

    sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue=hue, style=style, s=80)
    plt.xlabel(f'PC1 ({explained[0]:.1f}%)')
    plt.ylabel(f'PC2 ({explained[1]:.1f}%)')
    plt.title(f'PCA: {method_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pca_grid(results, metadata, hue="Strains", style="bio_rep"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec

    norm_names = sorted({k.split(" + ")[0] for k in results})
    scale_names = sorted({k.split(" + ")[1] for k in results})

    n_rows = len(norm_names)
    n_cols = len(scale_names)

    # Add 1 extra column for legends
    fig = plt.figure(figsize=(30, 28))
    spec = gridspec.GridSpec(n_rows, n_cols + 1, width_ratios=[1]*n_cols + [0.4], wspace=0.4, hspace=0.5)

    hue_vals = metadata[hue].unique()
    style_vals = metadata[style].unique()


    palette = dict(zip(hue_vals, sns.color_palette("hls", n_colors=len(hue_vals))))

    # Marker shapes for styles
    marker_shapes = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', 'H']
    style_marker_map = dict(zip(style_vals, marker_shapes))

    hue_handles = {
        val: Line2D([0], [0], marker='o', linestyle='None', markersize=8,
                    markerfacecolor=palette[val], markeredgecolor='black', label=val)
        for val in hue_vals
    }

    style_handles = {
        val: Line2D([0], [0], marker=style_marker_map[val], linestyle='None', markersize=8,
                    markerfacecolor='white', markeredgecolor='black', label=val)
        for val in style_vals
    }

    # Create subplots for PCA grid
    axes = []
    for i in range(n_rows):
        row_axes = []
        for j in range(n_cols):
            ax = fig.add_subplot(spec[i, j])
            row_axes.append(ax)
        axes.append(row_axes)

    # Plot PCA results
    for i, norm in enumerate(norm_names):
        for j, scale in enumerate(scale_names):
            label = f"{norm} + {scale}"
            ax = axes[i][j]

            if label in results:
                pca_data, explained = results[label]
                df_plot = pd.DataFrame({
                    "PC1": pca_data[:, 0],
                    "PC2": pca_data[:, 1],
                    "hue": metadata[hue].values,
                    "style": metadata[style].values
                })

                sns.scatterplot(
                    data=df_plot,
                    x="PC1",
                    y="PC2",
                    hue="hue",
                    style="style",
                    palette=palette,
                    markers=style_marker_map,
                    s=30,
                    ax=ax,
                    legend=False,
                    edgecolor='black',
                    linewidth=0.5
                )

                ax.set_title(label, fontsize=10)
                ax.tick_params(axis='both', labelsize=5)
                ax.tick_params(axis='x')
                if i < n_rows - 1:
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel(f'PC1 ({explained[0]:.1f}%)')

                if j > 0:
                    ax.set_ylabel("")
                else:
                    ax.set_ylabel(f'PC2 ({explained[1]:.1f}%)')

    # Create main legend handles
    hue_legend = list(hue_handles.values())
    style_legend = list(style_handles.values())

    # Place strains legend
    fig.legend(
        handles=hue_legend,
        title=hue.capitalize(),
        loc='upper right',
        bbox_to_anchor=(0.9, 0.95),
        fontsize='small',
        title_fontsize='medium',
        frameon=True
    )

    # Place bio_rep legend
    fig.legend(
        handles=style_legend,
        title=style.capitalize(),
        loc='lower right',
        bbox_to_anchor=(0.95, 0.5),
        fontsize='small',
        title_fontsize='medium',
        frameon=True
    )

    # Final layout adjustment
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()



# Create a dictionary of normalization methods and scaling methods
normalizations = {
    "None": normalize_none,
    "TIC": normalize_tic,
    "Max": normalize_max,
    "Log": normalize_log
}

scalings = {
    "None": scale_none,
    "Z-Score": scale_zscore,
    #"Min-Max": scale_minmax,
    #"Log": scale_log,
    "Pareto": scale_pareto,
    "Unit-Vector": scale_unit_vector,
    "MaxAbs": scale_maxabs,
    "Robust": scale_robust
}


# Apply each combo and run PCA
results = {}

for norm_name, norm_func in normalizations.items():
    for scale_name, scale_func in scalings.items():
        label = f"{norm_name} + {scale_name}"
        normed = norm_func(intensity_data.copy())
        scaled = scale_func(normed)

        # save the transformed matrix to CSV
        transformed_df = pd.DataFrame(
            scaled,
            columns=mz_values,
            index=sample_ids
        )
        transformed_df.insert(0, "SampleID", sample_ids)

        output_filename = f"{norm_name}_{scale_name}_12.csv".replace(" ", "_")
        transformed_df.to_csv(output_filename, index=False)

        # perform PCA
        pca_data, explained = perform_pca(scaled)
        results[label] = (pca_data, explained)


# plot
# for label, (pca_data, explained) in results.items():
#     plot_pca_with_metadata(pca_data, explained, label, meta_df)


# Plot the full grid
plot_pca_grid(results, meta_df, hue=hue_col, style=style_col)


pca_export = pd.DataFrame({"SampleID": sample_ids})

for label, (pca_data, _) in results.items():
    pca_export[f'{label}_PC1'] = pca_data[:, 0]
    pca_export[f'{label}_PC2'] = pca_data[:, 1]

pca_export.to_csv("PCA_all_combinations12.csv", index=False)