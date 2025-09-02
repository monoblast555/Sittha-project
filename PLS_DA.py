import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
import matplotlib.colors as mcolors
from matplotlib import colormaps
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('pca33_with_metadata.csv')

file = '33'
cl = 'Strains'
rep = 'exp_rep'

# Define Features and Labels
n_components = 50
pc_columns = [f'PC{i+1}' for i in range(n_components)]
X = data[pc_columns].values
y_labels = data[cl]
groups = data[rep]

# Encode Labels to Binary Matrix
lb = LabelBinarizer()
Y = lb.fit_transform(y_labels)

# Set Up Grouped K-Fold CV
gkf = GroupKFold(n_splits=4)

# Track performance across n_components values
results = []
best_score = 0
best_n_components = None
best_predictions = None
best_train_predictions = None


for n_components in range(2, 51):
    y_true_test_all, y_pred_test_all = [], []
    y_true_train_all, y_pred_train_all = [], []

    for train_idx, test_idx in gkf.split(X, Y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        n_used = min(n_components, X_train.shape[1], Y_train.shape[1])
        pls = PLSRegression(n_components=n_used)
        pls.fit(X_train, Y_train)

        # Test predictions
        Y_pred_test = pls.predict(X_test)
        y_pred_test = lb.inverse_transform(Y_pred_test)
        y_true_test = lb.inverse_transform(Y_test)
        y_pred_test_all.extend(y_pred_test)
        y_true_test_all.extend(y_true_test)

        # Train predictions
        Y_pred_train = pls.predict(X_train)
        y_pred_train = lb.inverse_transform(Y_pred_train)
        y_true_train = lb.inverse_transform(Y_train)
        y_pred_train_all.extend(y_pred_train)
        y_true_train_all.extend(y_true_train)

    acc = accuracy_score(y_true_test_all, y_pred_test_all)
    results.append((n_components, acc))

    if acc > best_score:
        best_score = acc
        best_n_components = n_components
        best_predictions = (y_true_test_all, y_pred_test_all)
        best_train_predictions = (y_true_train_all, y_pred_train_all)

# === Print tuning results ===
print("=== Accuracy by n_components ===")
for n, acc in results:
    print(f"Components: {n}, Accuracy: {acc:.3f}")
print(f"\nBest accuracy: {best_score:.3f} with {best_n_components} components\n")

# Final evaluation with best predictions


y_true_test, y_pred_test = best_predictions
y_true_train, y_pred_train = best_train_predictions

print("=== Classification Report (Test Set) ===")
print(classification_report(y_true_test, y_pred_test, zero_division=0))
print("=== Classification Report (Train Set) ===")
print(classification_report(y_true_train, y_pred_train, zero_division=0))

final_test_ccr = accuracy_score(y_true_test, y_pred_test)
final_train_ccr = accuracy_score(y_true_train, y_pred_train)
print(f"Test CCR (Accuracy): {final_test_ccr:.4f}")
print(f"Train CCR (Accuracy): {final_train_ccr:.4f}")

y_true_all, y_pred_all = best_predictions
print("=== Classification Report ===")
print(classification_report(y_true_all, y_pred_all, zero_division=0))
final_ccr = accuracy_score(y_true_all, y_pred_all)
print(f"Overall CCR (Correct Classification Rate): {final_ccr:.4f}")

# Compute confusion matrices
cm_train = confusion_matrix(y_true_train, y_pred_train, labels=lb.classes_)
cm_test = confusion_matrix(y_true_test, y_pred_test, labels=lb.classes_)
labels = lb.classes_

# Plot both confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Train CM
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title(f'Train Confusion Matrix\n(n_components = {best_n_components})')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# Test CM
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title(f'Test Confusion Matrix\n(n_components = {best_n_components})')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.suptitle(f'PLS-DA Train vs Test Confusion Matrices, {file} {cl} {rep}')
plt.tight_layout()
plt.show()



# Misclassified Samples
misclassified = [(true, pred) for true, pred in zip(y_true_all, y_pred_all) if true != pred]
print(f"\nTotal misclassified samples: {len(misclassified)}")
if misclassified:
    print("Some examples:")
    for i, (true, pred) in enumerate(misclassified[:10]):  # Show first 10
        print(f"{i+1}. True: {true}, Predicted: {pred}")

# Visualize PLS scores from best model

pls_final = PLSRegression(n_components=best_n_components)
pls_final.fit(X, Y)
X_scores = pls_final.x_scores_

score_df = pd.DataFrame(X_scores[:, :2], columns=['PLS1', 'PLS2'])
score_df[cl] = y_labels.values


strain_list = score_df[cl].unique()
n_strains = len(strain_list)
colors = colormaps['nipy_spectral'].resampled(n_strains)
palette = {strain: mcolors.rgb2hex(colors(i)) for i, strain in enumerate(strain_list)}

# Plot
plt.figure(figsize=(14, 12))
sns.scatterplot(
    data=score_df,
    x='PLS1',
    y='PLS2',
    hue=cl,
    palette=palette,
    s=60,
    edgecolor='black',
    linewidth=0.5
)


plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    title=cl,
    borderaxespad=0.,
    ncol=1,
    fontsize='small'
)
plt.title(f'PLS-DA Score Plot (Components 1 & 2, n={best_n_components}): {file} {cl} {rep} ')
plt.xlabel('PLS Component 1')
plt.ylabel('PLS Component 2')
plt.tight_layout()
plt.show()

print("\n=== Final Evaluation Summary ===")
print(f"Best number of components: {best_n_components}")
print(f"Overall CCR (Accuracy): {final_ccr:.4f}")
print(f"Total misclassified samples: {len(misclassified)}")

# Permutation Test
print("\n=== Permutation Test ===")

n_permutations = 1000
permuted_accuracies = []

np.random.seed(42)  # For reproducibility

for i in range(n_permutations):
    # Shuffle strain labels
    y_labels_permuted = np.random.permutation(y_labels)
    Y_permuted = lb.fit_transform(y_labels_permuted)

    y_true_perm_all, y_pred_perm_all = [], []

    for train_idx, test_idx in gkf.split(X, Y_permuted, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y_permuted[train_idx], Y_permuted[test_idx]

        n_used = min(best_n_components, X_train.shape[1], Y_train.shape[1])
        pls = PLSRegression(n_components=n_used)
        pls.fit(X_train, Y_train)

        Y_pred_test = pls.predict(X_test)
        y_pred_test = lb.inverse_transform(Y_pred_test)
        y_true_test = lb.inverse_transform(Y_test)

        y_pred_perm_all.extend(y_pred_test)
        y_true_perm_all.extend(y_true_test)

    perm_acc = accuracy_score(y_true_perm_all, y_pred_perm_all)
    permuted_accuracies.append(perm_acc)

    if (i + 1) % 100 == 0:
        print(f"Permutation {i+1}/{n_permutations} complete")

# Compute p-value
permuted_accuracies = np.array(permuted_accuracies)
p_value = np.mean(permuted_accuracies >= final_ccr)

print(f"\nPermutation test p-value: {p_value:.4f}")
print(f"Observed Accuracy: {final_ccr:.4f}")
print(f"Null Distribution Mean Accuracy: {np.mean(permuted_accuracies):.4f}")

# Plot null distribution
plt.figure(figsize=(10, 6))
sns.histplot(permuted_accuracies, bins=30, kde=True, color='gray', label='Null Distribution')
plt.axvline(final_ccr, color='red', linestyle='--', label=f'Observed Accuracy: {final_ccr:.4f}')
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.title("Permutation Test: Null Distribution of Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
