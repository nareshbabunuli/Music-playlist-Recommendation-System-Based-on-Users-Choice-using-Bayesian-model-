import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the data
data = pd.read_csv('data/data.csv')

# Preprocessing: Remove duplicates based on 'name' and 'artists' columns
data = data.drop_duplicates(subset=['name', 'artists'])

# Define popularity threshold
popularity_threshold = data['popularity'].median()

# Create a binary target variable: 1 if the song is popular, 0 otherwise
data['is_popular'] = (data['popularity'] > popularity_threshold).astype(int)

# Drop original popularity and other non-numeric columns for simplicity
data_cleaned = data.drop(columns=['popularity', 'id', 'name', 'artists', 'release_date'])

# Extract features and target
X = data_cleaned.drop('is_popular', axis=1)
y = data_cleaned['is_popular']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Summary
print("\nData Overview:")
print(data.head())
print(data.shape)
print(data.isnull().sum())

print("\nDetailed Data Summary:")

# Detailed Data Summary
print(
    "\nValence: Ranges from {} to {}, with a mean of approximately {:.2f}. Valence is a measure of musical positivity.".format(
        data['valence'].min(), data['valence'].max(), data['valence'].mean()))
print(
    "Year: The songs in the dataset span from {} to {}.".format(data['release_date'].min(), data['release_date'].max()))
print("Acousticness: Ranges from {} to {}, with an average value of {:.2f}.".format(data['acousticness'].min(),
                                                                                    data['acousticness'].max(),
                                                                                    data['acousticness'].mean()))
print("Danceability: Ranges from {} to {}, with a mean of approximately {:.2f}.".format(data['danceability'].min(),
                                                                                        data['danceability'].max(),
                                                                                        data['danceability'].mean()))
print("Duration: The average song duration is about {:.0f} milliseconds (or about {:.2f} minutes).".format(
    data['duration_ms'].mean(), data['duration_ms'].mean() / 60000))
print(
    "Energy: Ranges from {} to {}, with an average value of {:.2f}.".format(data['energy'].min(), data['energy'].max(),
                                                                            data['energy'].mean()))
print("Explicit: Most songs are not explicit (mean = {:.3f}).".format(data['explicit'].mean()))
print(
    "Instrumentalness: This feature shows a wide range, indicating a mix of songs with varying degrees of instrumentals.")
print("Key: Represents the key in which the song is composed and spans values from {} to {}.".format(data['key'].min(),
                                                                                                     data['key'].max()))
print(
    "Liveness: The average value is {:.3f}, suggesting that most tracks are studio recordings rather than live performances.".format(
        data['liveness'].mean()))
print("Loudness: Has a wide range, with an average loudness of {:.2f} decibels.".format(data['loudness'].mean()))
print("Mode: Majority of the songs are in the major mode (mean = {:.3f}).".format(data['mode'].mean()))
print(
    "Popularity: Ranges from {} to {}, with a mean value of {:.2f}. This might indicate that the dataset contains a mix of both popular and not-so-popular songs.".format(
        data['popularity'].min(), data['popularity'].max(), data['popularity'].mean()))
print("Speechiness: On average, songs have a speechiness value of {:.3f}.".format(data['speechiness'].mean()))
print("Tempo: Ranges widely, with an average tempo of about {:.2f} BPM.".format(data['tempo'].mean()))
# EDA: Distribution Visualization
features_to_visualize = ['danceability', 'energy', 'loudness', 'valence', 'acousticness']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_to_visualize, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# EDA: Correlation Analysis
correlation_matrix = data[features_to_visualize].corr()
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Key Features')
plt.show()

# Define and evaluate additional classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

for name, clf in classifiers.items():
    # Train classifier
    clf.fit(X_train, y_train)
    # Predict on test set
    y_pred = clf.predict(X_test)
    # Evaluate classifier
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Display evaluation metrics
    print(f"\nClassifier: {name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=['Not Popular', 'Popular'],
                yticklabels=['Not Popular', 'Popular'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({name})')
    plt.show()

# GridSearchCV for GaussianNB
param_grid = {
    'var_smoothing': np.logspace(-10, -5, 50)
}
grid_search = GridSearchCV(GaussianNB(), param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
# Best parameters
print("Best Parameters (GaussianNB): ", grid_search.best_params_)
# Predict on the test set with the optimized model
y_pred_nb = grid_search.predict(X_test)
# Evaluate the GaussianNB model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
# Display evaluation metrics for GaussianNB
print(f"\nClassifier: Gaussian Naive Bayes")
print(f"Accuracy: {accuracy_nb:.2f}")
print(f"Precision: {precision_nb:.2f}")
print(f"Recall: {recall_nb:.2f}")
print(f"F1 Score: {f1_nb:.2f}")
# Plot confusion matrix for Gaussian Naive Bayes classifier
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_nb, annot=True, fmt='g', cmap='Blues',
            xticklabels=['Not Popular', 'Popular'],
            yticklabels=['Not Popular', 'Popular'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Gaussian Naive Bayes)')
plt.show()


# Recommendation function
def recommend(artist_name, songs_data, classifier=grid_search.best_estimator_):
    # Filter songs by the specified artist
    artist_songs = songs_data[songs_data['artists'].str.contains(artist_name, case=False, na=False)]
    # Drop 'predicted_popular' column if it exists
    if 'predicted_popular' in artist_songs.columns:
        artist_songs = artist_songs.drop(columns=['predicted_popular'])
    # Extract features and predict popularity
    X_artist = artist_songs.drop(columns=['popularity', 'id', 'name', 'artists', 'release_date', 'is_popular'])
    artist_songs['predicted_popular'] = classifier.predict(X_artist)
    # Get songs predicted to be popular
    recommended_songs = artist_songs[artist_songs['predicted_popular'] == 1][['name']]
    return recommended_songs['name'].head(15).tolist()


# For Gaussian Naive Bayes recommendations
user_choice = 'Taylor Swift'
recommendations_nb = recommend('Taylor Swift', data)
print(recommendations_nb)
