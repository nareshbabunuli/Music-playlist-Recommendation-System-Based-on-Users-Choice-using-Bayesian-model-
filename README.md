
# Song Popularity Analysis and Music playlist Recommendation

This script provides an analysis of a dataset containing song details. It aims to predict the popularity of songs based on various features and recommends songs for a given artist that are predicted to be popular.

## Features:

1. **Data Loading**: Loads data from a CSV file named `data/data.csv`.
2. **Data Preprocessing**:
   - Removes duplicates based on the 'name' and 'artists' columns.
   - Creates a binary target variable, `is_popular`, based on the median popularity value.
   - Drops non-numeric columns for model training.
3. **Data Splitting**: Splits the data into training and testing sets.
4. **Data Overview**: Provides an overview of the data with summary statistics.
5. **Model Training and Evaluation**: Trains and evaluates several machine learning classifiers, including:
   - Logistic Regression
   - K-Nearest Neighbors
   - Decision Tree
   - AdaBoost
   - Gradient Boosting
   - Gaussian Naive Bayes
   Displays performance metrics such as accuracy, precision, recall, and F1 score, and plots confusion matrices for each model.
6. **Song Recommendation**: Provides song recommendations for a given artist using the best-performing model.

## Usage:

1. Ensure you have all the required libraries installed.
2. Load your dataset in the `data/` directory with the filename `data.csv`.
3. Run the script to perform analysis and model training.
4. Use the `recommend` function to get song recommendations for your desired artist.

## Dependencies:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
