import pandas as pd

# Load the dataset
data = pd.read_csv("data/data.csv")  # Please replace with the correct path

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Display basic statistics for each column
print(data.describe(include='all'))

# List of attributes you might want to describe
attributes = ['danceability', 'energy', 'loudness', 'valence', 'acousticness']  # Add other columns as needed

# Displaying descriptions for selected attributes
for attr in attributes:
    print(f"{attr}:")
    print(data[attr].describe())
    print("\n")
