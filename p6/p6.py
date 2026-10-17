# Practical 6: K-Means Clustering (Vehicle Registration Example)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Creating a sample Vehicle Dataset
data = {
    'Registration_No': ['MH12AB1234', 'MH14CD5678', 'DL01EF9012', 'KA05GH3456',
                        'MH12IJ7890', 'DL02KL2345', 'KA03MN6789', 'MH15OP1234',
                        'DL03QR5678', 'KA01ST9012', 'MH12UV3456', 'DL04WX7890'],
    'Age_Years': [2, 5, 8, 3, 4, 10, 6, 1, 9, 7, 3, 11],
    'Mileage_kmpl': [18, 14, 12, 17, 16, 10, 13, 20, 11, 12, 18, 9],
    'Engine_CC': [1200, 1500, 2000, 1100, 1400, 2200, 1600, 1000, 1800, 1900, 1300, 2500]
}

df = pd.DataFrame(data)
print("Vehicle Dataset:")
print(df)

# Features for clustering
X = df[['Age_Years', 'Mileage_kmpl', 'Engine_CC']]

# Applying K-Means with K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

print("\nCluster Assigned to each Vehicle:")
print(df[['Registration_No', 'Age_Years', 'Mileage_kmpl', 'Engine_CC', 'Cluster']])

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Visualization (using Age vs Mileage)
plt.figure(figsize=(8, 5))
colors = ['red', 'blue', 'green']

for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Age_Years'], cluster_data['Mileage_kmpl'],
                color=colors[i], label=f'Cluster {i}', s=100)

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            color='black', marker='X', s=200, label='Centroids')

plt.xlabel('Vehicle Age (Years)')
plt.ylabel('Mileage (kmpl)')
plt.title('K-Means Clustering of Vehicles')
plt.legend()
plt.grid(True)
plt.show()
