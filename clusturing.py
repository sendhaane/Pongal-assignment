from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


data = pd.DataFrame({
    'Latitude': np.random.uniform(10, 20, 100),  # Random latitudes
    'Longitude': np.random.uniform(30, 40, 100)  # Random longitudes
})

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['Latitude', 'Longitude']])

# Plotting clusters
plt.scatter(data['Latitude'], data['Longitude'], c=data['Cluster'], cmap='viridis')
plt.title('Accident Hotspots')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()
