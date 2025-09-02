import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from itertools import combinations
from matplotlib import colormaps
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Load data
df = pd.read_csv('pca33_with_metadata.csv')
file = '33'
cl = 'Strains'
repi = 'exp_rep'

# Extract labels and replicates
labels = df[cl]
rep = df[repi]

# Combine all data into a single DataFrame
data = df.copy()
data[repi] = rep
data[cl] = labels

# Get unique replicates
unique_reps = sorted(data[repi].unique())
rep_combinations = list(combinations(unique_reps, 3))  # All combinations of 3 for training

best_pc_count = None
best_accuracy = 0
accuracy_per_pc = {}

# Find the best number of PCs
for pc_count in range(2, 51):  # PC1 to PC50
    test_accuracies = []

    for train_reps in rep_combinations:
        test_rep = list(set(unique_reps) - set(train_reps))[0]

        train_data = data[data[repi].isin(train_reps)]
        test_data = data[data[repi] == test_rep]

        X_train = train_data.iloc[:, 1:1+pc_count]
        y_train = train_data[cl]
        X_test = test_data.iloc[:, 1:1+pc_count]
        y_test = test_data[cl]

        lda = LDA()
        lda.fit(X_train, y_train)

        y_test_pred = lda.predict(X_test)
        acc = accuracy_score(y_test, y_test_pred)
        test_accuracies.append(acc)

    avg_acc = np.mean(test_accuracies)
    accuracy_per_pc[pc_count] = avg_acc

    if avg_acc > best_accuracy:
        best_accuracy = avg_acc
        best_pc_count = pc_count

print(f"\n Best PC Count: {best_pc_count} (Average Test Accuracy: {best_accuracy:.4f})")

# Plot performance across PC counts
plt.figure(figsize=(10, 5))
plt.plot(list(accuracy_per_pc.keys()), list(accuracy_per_pc.values()), marker='o')
plt.axvline(best_pc_count, color='red', linestyle='--', label=f'Best PC Count: {best_pc_count}')
plt.title("Average Test Accuracy vs. Number of Principal Components")
plt.xlabel("Number of PCs")
plt.ylabel("Average Test Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# LDA projection of all data using the best number of PCs


# Prepare the full dataset
X_all = data.iloc[:, 1:1+best_pc_count]
y_all = data[cl]

# Fit and transform with LDA
lda_all = LDA(n_components=2)
X_all_lda = lda_all.fit_transform(X_all, y_all)

# Plot LDA

strain_labels = data[cl].unique()
num_classes = len(strain_labels)
colors = colormaps['nipy_spectral'].resampled(num_classes)
palette = {strain: mcolors.rgb2hex(colors(i)) for i, strain in enumerate(strain_labels)}


plt.figure(figsize=(14, 12))
for label in strain_labels:
    idx = (y_all == label)
    plt.scatter(
        X_all_lda[idx, 0],
        X_all_lda[idx, 1],
        label=label,
        alpha=0.7,
        color=palette[label]
    )

plt.title(f"LDA plot (Best PC Count = {best_pc_count}) : {file} {cl} {repi}")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    title=cl,
    borderaxespad=0.,
    ncol=1,
    fontsize='small'
)
plt.grid(True)
plt.tight_layout()
plt.show()

# Final evaluation using best PC count

all_y_true = []
all_y_pred = []
all_train_y_true = []
all_train_y_pred = []

for train_reps in rep_combinations:
    test_rep = list(set(unique_reps) - set(train_reps))[0]
    print(f"\n=== Final Evaluation | Train Reps: {train_reps}, Test Rep: {test_rep} ===")

    train_data = data[data[repi].isin(train_reps)]
    test_data = data[data[repi] == test_rep]

    X_train = train_data.iloc[:, 1:1+best_pc_count]
    y_train = train_data[cl]
    X_test = test_data.iloc[:, 1:1+best_pc_count]
    y_test = test_data[cl]

    lda = LDA(n_components=2)
    lda.fit(X_train, y_train)

    # Predict and store test results
    y_pred = lda.predict(X_test)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    # Predict and store training results
    y_train_pred = lda.predict(X_train)
    all_train_y_true.extend(y_train)
    all_train_y_pred.extend(y_train_pred)

    # Classification Report
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))

# Compute confusion matrices
cm_test = confusion_matrix(all_y_true, all_y_pred)
cm_train = confusion_matrix(all_train_y_true, all_train_y_pred)

# Plot both training and test confusion matrices
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Training CM
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=lda.classes_)
disp_train.plot(ax=axs[0], cmap='Greens', xticks_rotation=45)
axs[0].set_title("Training Confusion Matrix")

# Testing CM
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=lda.classes_)
disp_test.plot(ax=axs[1], cmap='Blues', xticks_rotation=45)
axs[1].set_title("Test Confusion Matrix")

plt.suptitle(f"LDA Train vs Test Confusion Matrices: {file} {cl} {repi}")
plt.tight_layout()
plt.show()

# Overall CCR (Correct Classification Rate)
overall_ccr = accuracy_score(all_y_true, all_y_pred)
print(f"\n=== Overall CCR (LDA, all test predictions) ===")
print(f"Overall CCR (Correct Classification Rate): {overall_ccr:.4f}")

# Final classification report
print("\nOverall Classification Report (LDA):")
print(classification_report(all_y_true, all_y_pred, zero_division=0))

# Permutation Test
print("\n=== Permutation Test ===")

num_permutations = 1000
permuted_accuracies = []

np.random.seed(42)  # For reproducibility

for i in range(num_permutations):
    permuted_y = data[cl].sample(frac=1, replace=False).reset_index(drop=True)
    data_permuted = data.copy()
    data_permuted[cl] = permuted_y

    permuted_y_true = []
    permuted_y_pred = []

    for train_reps in rep_combinations:
        test_rep = list(set(unique_reps) - set(train_reps))[0]

        train_data = data_permuted[data_permuted[repi].isin(train_reps)]
        test_data = data_permuted[data_permuted[repi] == test_rep]

        X_train = train_data.iloc[:, 1:1+best_pc_count]
        y_train = train_data[cl]
        X_test = test_data.iloc[:, 1:1+best_pc_count]
        y_test = test_data[cl]

        lda = LDA()
        lda.fit(X_train, y_train)

        y_pred = lda.predict(X_test)

        permuted_y_true.extend(y_test)
        permuted_y_pred.extend(y_pred)

    acc = accuracy_score(permuted_y_true, permuted_y_pred)
    permuted_accuracies.append(acc)

    if (i + 1) % 100 == 0:
        print(f"Permutation {i+1}/{num_permutations} complete")

# Compute p-value (proportion of permuted accuracies >= observed accuracy)
p_value = np.mean(np.array(permuted_accuracies) >= overall_ccr)
print(f"\nPermutation test p-value: {p_value:.4f}")

# Plot the null distribution
plt.figure(figsize=(10, 6))
sns.histplot(permuted_accuracies, bins=30, kde=True, color='gray', label='Null Distribution')
plt.axvline(overall_ccr, color='red', linestyle='--', label=f'Observed Accuracy: {overall_ccr:.4f}')
plt.title("Permutation Test: Null Distribution of Accuracies")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
