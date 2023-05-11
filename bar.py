# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:08:31 2023

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
