# Practical 5: K-Nearest Neighbors (KNN) Classification

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Compare different values of K
k_values = [1, 3, 5, 7, 9]
accuracies = []

print("\nAccuracy for different values of K:")
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K = {k} --> Accuracy = {acc:.4f}")

# Best K
best_k = k_values[np.argmax(accuracies)]
print(f"\nBest value of K: {best_k}")

# Detailed results for best K
best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=iris.target_names))

# Plot Accuracy vs K
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy for Different Values of K')
plt.xticks(k_values)
plt.grid(True)
plt.show()