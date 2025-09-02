import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import permutation_test_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import PredefinedSplit

# Load CSV data
df = pd.read_csv('pca33_with_metadata.csv')

file = '33'
cl = 'Strains'
rep = 'exp_rep'

# Select features and target
X_all = df.iloc[:, 1:51]  # PC1 to PC50
y = df[cl]

# PC Selection using cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for n_pcs in range(2, 51):
    X = X_all.iloc[:, :n_pcs]
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42
    )
    scores = cross_val_score(rf, X, y, cv=cv)
    mean_score = scores.mean()
    results.append((n_pcs, mean_score))
    print(f"{n_pcs} PCs -> Mean CV accuracy: {mean_score:.4f}")

# Best number of PCs
best_n, best_score = max(results, key=lambda x: x[1])
print(f"\nBest number of PCs: {best_n} (CV accuracy = {best_score:.4f})")



# Replicate-based training/testing

unique_reps = df[rep].unique()
all_y_true = []
all_y_pred = []
all_train_y_true = []
all_train_y_pred = []

for test_rep in unique_reps:
    test_mask = (df[rep] == test_rep)
    train_mask = ~test_mask

    X_train = X_all.loc[train_mask].iloc[:, :best_n]
    y_train = y[train_mask]
    X_test = X_all.loc[test_mask].iloc[:, :best_n]
    y_test = y[test_mask]

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Predict and store test results
    y_pred = rf.predict(X_test)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    # Predict and store training results
    y_train_pred = rf.predict(X_train)
    all_train_y_true.extend(y_train)
    all_train_y_pred.extend(y_train_pred)

    print(f"\nTest Replicate: {test_rep}")
    print(classification_report(y_test, y_pred, zero_division=0))



# Final overall evaluation
print("\n=== Overall Classification Report ===")
print(classification_report(all_y_true, all_y_pred, zero_division=0))


overall_ccr = accuracy_score(all_y_true, all_y_pred)
print(f"\nOverall CCR (Correct Classification Rate): {overall_ccr:.4f}")

# Compute confusion matrices
cm_test = confusion_matrix(all_y_true, all_y_pred)
cm_train = confusion_matrix(all_train_y_true, all_train_y_pred)

# Plot both training and test confusion matrices
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Training CM
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=rf.classes_)
disp_train.plot(ax=axs[0], cmap='Greens', xticks_rotation=45)
axs[0].set_title("Training Confusion Matrix")

# Testing CM
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=rf.classes_)
disp_test.plot(ax=axs[1], cmap='Blues', xticks_rotation=45)
axs[1].set_title("Test Confusion Matrix")

plt.suptitle(f"Random Forest Train vs Test Confusion Matrices, {file} {cl} {rep}")
plt.tight_layout()
plt.show()



# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Select optimal PC features
X = X_all.iloc[:, :best_n]

# Create PredefinedSplit based on replicates
unique_reps = df[rep].unique()
rep_to_fold = {rep_name: idx for idx, rep_name in enumerate(unique_reps)}
fold_indices = df[rep].map(rep_to_fold).values
ps = PredefinedSplit(test_fold=fold_indices)

# Define the classifier
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)

# Run the permutation test
print("\n=== Permutation Test (Replicate-Based CV) ===")
score, perm_scores, p_value = permutation_test_score(
    rf, X, y_encoded,
    cv=ps,
    scoring="accuracy",
    n_permutations=1000,
    n_jobs=-1,
    random_state=42
)

# Output results
print(f"Observed Accuracy (Replicate-Based CV): {score:.4f}")
print(f"p-value from permutation test: {p_value:.4f}")

# Plot the permutation distribution
plt.figure(figsize=(8, 5))
plt.hist(perm_scores, bins=30, alpha=0.7, color='skyblue')
plt.axvline(score, color='red', linestyle='--', label='Observed Accuracy')
plt.title('Permutation Test (Replicate-Based CV)')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
