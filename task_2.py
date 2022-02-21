import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv('Iris.csv')
X = data.iloc[:, 1:5].values
print(X)

# Find the optimum number of cluster
# WCSS stand for within cluster sum of square
wcss = []
for i in range(1,15):
    k_means = KMeans(n_clusters=i, init='k-means++', random_state=0)
    k_means.fit(X)
    wcss.append(k_means.inertia_)

# Plot to show optimal k using Elbow method
numOfCluster = range(1,15)
plt.plot(numOfCluster,wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Wcss')
plt.show()

# Build model
# when we plot the graph we see that the optimal value for k is 3
k_means_model = KMeans(n_clusters=3, init='k-means++', random_state=0)
predicted_K_means = k_means_model.fit_predict(X)


# Plot clusters
plt.scatter(X[predicted_K_means == 0, 0], X[predicted_K_means == 0, 1], c ='red', label = 'Iris-setosa')
plt.scatter(X[predicted_K_means == 1, 0], X[predicted_K_means == 1, 1], c ='blue', label = 'Iris-versicolour')
plt.scatter(X[predicted_K_means == 2, 0], X[predicted_K_means == 2, 1], c ='green', label = 'Iris-virginica')
plt.scatter(k_means_model.cluster_centers_[:, 0], k_means_model.cluster_centers_[:, 1], c ='black', label = 'Centroids')
plt.show()

