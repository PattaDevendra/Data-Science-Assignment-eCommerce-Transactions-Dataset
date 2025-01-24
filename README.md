# Data-Science-Assignment-eCommerce-Transactions-Dataset
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt


customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')


data = pd.merge(customers, transactions, on="CustomerID")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))


inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)


plt.plot(range(2, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=4)  # You can choose k=4 or any value from your elbow method
clusters = kmeans.fit_predict(scaled_data)


db_index = davies_bouldin_score(scaled_data, clusters)
silhouette = silhouette_score(scaled_data, clusters)


pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca_components[:, 0], pca_components[:, 1], c=clusters, cmap='viridis')
plt.title('Customer Segments')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter)
plt.show()

print(f"DB Index: {db_index}")
print(f"Silhouette Score: {silhouette}")

centroids = kmeans.cluster_centers_
print("Cluster Centroids:", centroids)
