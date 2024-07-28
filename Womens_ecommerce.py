# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv('/Womens Clothing E-Commerce Reviews.csv')  # Adjust the path as needed

# Inspect the dataset
print("Dataset Info:")
print(data.info())

print("\nMissing Values Count:")
print(data.isnull().sum().sort_values(ascending=False))

# Drop missing values
data_cleaned = data.dropna()

# Basic statistics
print("\nData Description:")
print(data_cleaned.describe())

# Grouping and summing ratings by Clothing ID and Class Name
purchase_df = (data_cleaned.groupby(['Clothing ID', 'Class Name'])['Rating']
               .sum().unstack().reset_index().fillna(0).set_index('Clothing ID'))

print("\nPurchase DataFrame (Sample):")
print(purchase_df.head())

# Define a function to encode units
def encode_units(x):
    if x < 1:  # If the rating is less than 1
        return 0  # Not purchased
    if x >= 1:  # If the rating is 1 or greater
        return 1  # Purchased

# Apply encoding
purchase_df = purchase_df.applymap(encode_units)
print("\nEncoded Purchase DataFrame (Sample):")
print(purchase_df.head())

# Compute cosine similarities between clothing items
user_similarities = cosine_similarity(purchase_df)
user_similarity_data = pd.DataFrame(user_similarities, index=purchase_df.index, columns=purchase_df.index)

print("\nUser Similarity DataFrame (Sample):")
print(user_similarity_data.head())
