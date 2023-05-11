# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:28:46 2023

@author: Akshay
"""

import pandas as pd

# Read the dataset
dfn1 = pd.read_csv('API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_5359510.csv', skiprows=4)

# Display the head of the dataset
dfn1.head()


# Display the shape of the dataset
dfn1.shape


import numpy as np

# Load the dataset
df = pd.read_csv("API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_5359510.csv", skiprows=4)

# Select relevant columns and drop missing values
df = df[["Country Name", "2021"]].dropna()

# Rename columns
df.columns = ["Country", "Agriculture, forestry, and fishing, value added (% of GDP)"]

# Set index to country name
df.set_index("Country", inplace=True)

# Remove rows with invalid  values (negative or zero)
df = df[df["Agriculture, forestry, and fishing, value added (% of GDP)"] > 0]

# Log-transform the values to reduce skewness
df["Agriculture, forestry, and fishing, value added (% of GDP)"] = np.log(df["Agriculture, forestry, and fishing, value added (% of GDP)"])

# Standardize the data using z-score normalization
df = (df - df.mean()) / df.std()

# Save the cleaned dataset to a new file
df.to_csv("cluster_dataset.csv")

import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("cluster_dataset.csv")

# Select a sample of four countries
countries = ["China", "India", "Indonesia", "Argentina"]
sample_data = df.loc[df["Country"].isin(countries)]

# Reset index for plotting
sample_data.reset_index(drop=True, inplace=True)

# Plot the agricultural value added for the sample countries
plt.figure(figsize=(12, 8))
plt.bar(sample_data["Country"], sample_data["Agriculture, forestry, and fishing, value added (% of GDP)"])
plt.xlabel("Country", fontsize=14)
plt.ylabel("Agriculture Value Added", fontsize=14)
plt.title("Agriculture Value Added for Sample Countries", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="y")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the cleaned dataset
df = pd.read_csv("cluster_dataset.csv")

# Extract the interest column and normalize the data
X = df['Agriculture, forestry, and fishing, value added (% of GDP)'].values.reshape(-1, 1)
X_norm = (X - X.mean()) / X.std()

# Define the range of number of clusters to try
n_clusters_range = range(2, 11)

# Iterate over the number of clusters and compute the silhouette score
silhouette_scores = []
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X_norm)
    silhouette_scores.append(silhouette_score(X_norm, labels))

# Plot the silhouette scores
plt.plot(n_clusters_range, silhouette_scores)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis: Optimal Number of Clusters")
plt.show()


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load the cleaned dataset
df = pd.read_csv('cluster_dataset.csv')

# Extract Agriculture value added column and normalize
X = df['Agriculture, forestry, and fishing, value added (% of GDP)'].values.reshape(-1, 1)
X_norm = StandardScaler().fit_transform(X)

# Perform Gaussian Mixture Model clustering with n_clusters=4
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X_norm)
df['Cluster'] = gmm.predict(X_norm)

# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['red', 'green', 'blue', 'orange']
for i in range(4):
    cluster_data = df[df['Cluster'] == i]
    scatter = ax.scatter(cluster_data.index, cluster_data['Agriculture, forestry, and fishing, value added (% of GDP)'],
                         color=colors[i], label=f'Cluster {i+1}')
plt.xticks(np.arange(0, df.shape[0], 50), np.arange(0, df.shape[0], 50), fontsize=12)
plt.xlabel('Country Index', fontsize=14)
plt.ylabel('Agriculture Value Added (% of GDP)', fontsize=14)
plt.title('Gaussian Mixture Model Clustering Results', fontsize=16)
ax.legend(fontsize=12)

# Add annotation for the cluster centers
centers = gmm.means_
for i, center in enumerate(centers):
    ax.annotate(f'Cluster {i+1} center: {center[0]:,.2f}', xy=(1, center[0]), xytext=(6, 0),
                textcoords="offset points", ha='left', va='center', fontsize=12, color=colors[i])

plt.show()


# print countries in each cluster in a table
for i in range(4):
    print(f'Cluster {i+1}:')
    cluster_data = df[df['Cluster']==i]
    cluster_table = pd.DataFrame({'Country': cluster_data['Country'].values})
    display(cluster_table)
