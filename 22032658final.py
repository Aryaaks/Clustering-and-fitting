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


import pandas as pd

# Read the dataset
dfN2 = pd.read_csv('API_AG.LND.AGRI.K2_DS2_en_csv_v2_5362183.csv', skiprows=4)

# Display the head of the dataset
dfN2.head()



# Display the shape of the dataset
dfN2.shape


import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('API_AG.LND.AGRI.K2_DS2_en_csv_v2_5362183.csv', skiprows=4)

# Select only the necessary data for fitting analysis
df = df[['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', *df.columns[-32:-1]]]

# Rename columns to simpler names
df.columns = ['Country', 'Code', 'Indicator', 'IndicatorCode', *range(1990, 2021)]

# Melt the DataFrame to transform the columns into rows
df_melted = pd.melt(df, id_vars=['Country', 'Code', 'Indicator', 'IndicatorCode'], var_name='Year', value_name='Value')

# Drop rows with missing values
df_cleaned = df_melted.dropna()

# Save the cleaned data to a new CSV file
df_cleaned.to_csv('fitting_data.csv', index=False)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from errors import err_ranges
# Load the cleaned dataset
df = pd.read_csv('fitting_data.csv')

# Filter data for Nigeria
India_data = df[df['Country'] == 'India']

# Extract the necessary columns
years = India_data['Year'].values
values = India_data['Value'].values

# Fit a polynomial curve to the data
coeffs = np.polyfit(years, values, deg=2)
poly_func = np.poly1d(coeffs)

# Calculate the residuals
residuals = values - poly_func(years)

# Calculate the standard deviation of the residuals
std_dev = np.std(residuals)

# Generate predictions for future years
future_years = np.arange(years.min(), years.max() + 21) 
predicted_values = poly_func(future_years)

# Calculate upper and lower confidence bounds
upper_bound = predicted_values + 2 * std_dev
lower_bound = predicted_values - 2 * std_dev

# Plot the best fitting function and confidence range
plt.figure(figsize=(12, 8))
plt.plot(years, values, 'ko', label='Actual Data')
plt.plot(future_years, predicted_values, 'r-', label='Line of best fit')
plt.fill_between(future_years, lower_bound, upper_bound, color='blue', alpha=0.4, label='Confidence Range')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Agricultural land (sq. km)', fontsize=14)
plt.title('Polynomial Model for India', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()


