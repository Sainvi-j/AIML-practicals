# Practical 7: Principal Component Analysis (PCA)

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

print("Original Dataset Shape:", X.shape)
print("Features:", feature_names)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA - Reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nShape after PCA:", X_pca.shape)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Retained:")
print(round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

# Create DataFrame for results
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Target'] = y

print("\nFirst 10 rows of PCA transformed data:")
print(pca_df.head(10))

# Visualization
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
labels = iris.target_names

for i, label in enumerate(labels):
    plt.scatter(pca_df[pca_df['Target'] == i]['PC1'],
                pca_df[pca_df['Target'] == i]['PC2'],
                color=colors[i], label=label, s=50)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Iris Dataset (4D to 2D)')
plt.legend()
plt.grid(True)
plt.show()