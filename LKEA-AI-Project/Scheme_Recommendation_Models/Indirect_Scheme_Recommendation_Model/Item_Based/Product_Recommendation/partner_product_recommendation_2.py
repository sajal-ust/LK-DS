# -*- coding: utf-8 -*-
"""Partner_Product_Recommendation  2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XQUjajU7x6cf8lvVOYjCn6YX2DZbG4_l
"""

import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("stockist_data.csv")

# Define product columns (assumed to be in binary form)
product_cols = [
    "AIS(Air Insulated Switchgear)", "RMU(Ring Main Unit)", "PSS(Compact Sub-Stations)",
    "VCU(Vacuum Contactor Units)", "E-House", "VCB(Vacuum Circuit Breaker)",
    "ACB(Air Circuit Breaker)", "MCCB(Moduled Case Circuit Breaker)",
    "SDF(Switch Disconnectors)", "BBT(Busbar Trunking)", "Modular Switches"
]

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save train and test datasets to CSV
train_df.to_csv("train_data.csv", index=False)  # Save training data
test_df.to_csv("test_data.csv", index=False)    # Save testing data

print("Train and test data saved successfully.")

# Convert product purchase data to True/False format (needed because Jaccard similarity works with binary data)
df_products_train = train_df[product_cols].astype(bool)

# Compute Jaccard Similarity between products
# Convert the DataFrame to a NumPy array for faster computation
df_products_np = df_products_train.values
# Calculate Jaccard similarity between products (columns) based on purchase patterns
# '1 - distance' is used because pairwise_distances gives dissimilarity by default
jaccard_product_sim = 1 - pairwise_distances(df_products_np.T, metric="jaccard")

df_products_np

# Convert to DataFrame for better readability
# Create a DataFrame from the Jaccard similarity matrix
# Set product names as row and column labels for easy interpretation
product_similarity_df = pd.DataFrame(jaccard_product_sim, index=product_cols, columns=product_cols)
# Display the product similarity matrix
print(product_similarity_df)

# Function to get top 3 similar products for a given product
# Check if the product exists in the similarity matrix
def get_top3_products(product_name):
    """Returns top 3 most similar products for a given product using Jaccard similarity."""
    if product_name not in product_similarity_df.index:
        return ["Product not found"]
       # Sort products by similarity in descending order and return the top 3 (excluding itself)
    return list(product_similarity_df[product_name].sort_values(ascending=False)[1:4].index)

# Recommend products for each partner in the test set
# To store recommended product lists for each partner,recommendations
# To store corresponding similarity scores,similarity_scores
recommendations = []
similarity_scores = []

for index, row in test_df.iterrows():
    # Get the list of products that the partner has already purchased
    purchased_products = [product for product in product_cols if row[product] == 1]  # Products bought by partner

    # If no purchases, skip recommendation for this partner
    if not purchased_products:
        recommendations.append([])
        similarity_scores.append([])
        continue

    recommended_products = set()# Use a set to avoid duplicates
    product_scores = [] # Store similarity scores of recommended products

    for product in purchased_products:
        # Get top 3 similar products for each purchased product
        top_products = get_top3_products(product)
        recommended_products.update(top_products) # Add them to the recommendation set
        # Get similarity scores for those top products
        scores = product_similarity_df.loc[product, top_products].values
        product_scores.extend(scores)

    # Store exactly 3 recommendations and their similarity scores
    recommendations.append(list(recommended_products)[:3] if len(recommended_products) >= 3 else list(recommended_products))
    # Store the corresponding similarity scores for those recommended products
    similarity_scores.append(product_scores[:3] if len(product_scores) >= 3 else product_scores)

    # Display intermediate results for first 5 partners
    if index < 5:
        print(f"\nRecommendations for Partner {row['Partner_id']}:")
        print(f"Purchased Products: {purchased_products}")
        print(f"Recommended Products: {recommendations[-1]}")
        print(f"Similarity Scores: {similarity_scores[-1]}")

# Save recommendations in DataFrame
test_df["Recommended_Products"] = recommendations
test_df["Similarity_Scores"] = similarity_scores
recommended_df = test_df[["Partner_id", "Recommended_Products", "Similarity_Scores"]]

# Display final recommendation results
print("\nFinal Partner Product Recommendations:")
print(recommended_df.head())

# Save recommendations
output_file = "Partner_Product_Recommendations.csv"
recommended_df.to_csv(output_file, index=False)

print(f"\nProduct recommendations saved to {output_file}")



