import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("data/data.csv")  # Replace with the correct path

# Setting a style for seaborn plots
sns.set_style("whitegrid")

# 1. Histograms for each feature
for column in data.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=30, kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# 2. Box Plots for each feature
for column in data.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.show()

# 3. Scatter Plots for some pairs of features (you can customize the pairs as needed)
# Example: Danceability vs. Energy
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Danceability', y='Energy')
plt.title('Scatter Plot of Danceability vs Energy')
plt.show()

# 4. Correlation Matrix and Heatmaps
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# 5. Pair Plots for selected features (this might take a while for large datasets)
# Here, I'm taking a subset of features for illustration. You can customize as needed.
subset_features = ['Danceability', 'Energy']  # Add more features as needed
sns.pairplot(data[subset_features])
plt.suptitle('Pair Plots of Selected Features', y=1.02)
plt.show()
