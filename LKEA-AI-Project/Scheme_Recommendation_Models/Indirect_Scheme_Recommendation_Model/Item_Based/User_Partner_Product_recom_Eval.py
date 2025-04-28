# Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

def run_user_based_recommendation(include_purchased=True, is_lambda=False):
    """User-Based Recommendation Code with Similarity Scores"""

    # Load the dataset
    df = pd.read_csv("stockist_data.csv")

    # Define the product columns
    product_columns = [
        "AIS(Air Insulated Switchgear)", "RMU(Ring Main Unit)", "PSS(Compact Sub-Stations)",
        "VCU(Vacuum Contactor Units)", "E-House", "VCB(Vacuum Circuit Breaker)",
        "ACB(Air Circuit Breaker)", "MCCB(Moduled Case Circuit Breaker)",
        "SDF(Switch Disconnectors)", "BBT(Busbar Trunking)", "Modular Switches"
    ]

    # Extract user-product matrix and convert to binary
    user_product_matrix = df.set_index("Partner_id")[product_columns].astype(int)

    # Save the user IDs separately
    user_ids = user_product_matrix.index.tolist()

    # Split data into train and test sets (80% train, 20% test)
    train_data, test_data = train_test_split(user_product_matrix, test_size=0.2, random_state=42)

    # Save to CSV
    train_data.reset_index().to_csv("train_data.csv", index=False)
    test_data.reset_index().to_csv("test_data.csv", index=False)

    # Fit KNN model on training data
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(train_data)

    train_user_ids = train_data.index.tolist()

    # Fit KNN Model for User-Based Collaborative Filtering on training data
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(train_data)

    recommendations = []

    for partner_id in test_data.index:
        partner_vector = test_data.loc[[partner_id]]  # Pass as DataFrame with column names
        distances, indices = model.kneighbors(partner_vector, n_neighbors=6)

        recommended_products = []
        similarity_scores = []

        for idx, dist in zip(indices.flatten(), distances.flatten()):
            if idx == 0:
                continue  # skip self

            neighbor_id = train_user_ids[idx]
            neighbor_products = set(train_data.loc[neighbor_id][train_data.loc[neighbor_id] == 1].index)

            if include_purchased:
                recommended_products.extend(list(neighbor_products))
                similarity_scores.extend([1 - dist] * len(neighbor_products))
            else:
                purchased_products = set(test_data.loc[partner_id][test_data.loc[partner_id] == 1].index)
                filtered_products = list(neighbor_products - purchased_products)
                recommended_products.extend(filtered_products)
                similarity_scores.extend([1 - dist] * len(filtered_products))

        # De-duplicate products but keep the first occurrence's score
        product_score_pairs = dict()
        for product, score in zip(recommended_products, similarity_scores):
            if product not in product_score_pairs:
                product_score_pairs[product] = score

        top3_products = list(product_score_pairs.keys())[:3]
        top3_scores = [product_score_pairs[prod] for prod in top3_products]

        recommendations.append({
            "Partner_id": partner_id,
            "Recommended_Products": top3_products,
            "Similarity_Scores": top3_scores
        })

    # Save recommendations
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv("User_Based_Recommendations.csv", index=False)
    print("Recommendations saved to 'User_Based_Recommendations.csv'")
