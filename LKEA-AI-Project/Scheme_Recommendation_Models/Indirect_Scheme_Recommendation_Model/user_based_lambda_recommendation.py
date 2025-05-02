# user_based_lambda_recommendation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from indirect_lambda_handler import save_file
import os
import ast

PRODUCT_COLUMNS = [
    "AIS(Air Insulated Switchgear)", "RMU(Ring Main Unit)", "PSS(Compact Sub-Stations)",
    "VCU(Vacuum Contactor Units)", "E-House", "VCB(Vacuum Circuit Breaker)",
    "ACB(Air Circuit Breaker)", "MCCB(Moduled Case Circuit Breaker)",
    "SDF(Switch Disconnectors)", "BBT(Busbar Trunking)", "Modular Switches"
]

def run_user_based_recommendation(df, include_purchased=True):
    """Lambda-safe User-Based Recommendation Code"""

    # Extract user-product matrix
    user_product_matrix = df.set_index("Partner_id")[PRODUCT_COLUMNS].astype(int)
    user_ids = user_product_matrix.index.tolist()

    # Train/test split
    train_data, test_data = train_test_split(user_product_matrix, test_size=0.2, random_state=42)

    # Fit KNN model on training data
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(train_data)
    train_user_ids = train_data.index.tolist()

    recommendations = []

    for partner_id in test_data.index:
        partner_vector = test_data.loc[[partner_id]]
        distances, indices = model.kneighbors(partner_vector, n_neighbors=6)

        recommended_products = []
        similarity_scores = []

        for idx, dist in zip(indices.flatten(), distances.flatten()):
            if idx == 0:
                continue

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

        # Remove duplicates, keep first score
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

    # Convert to DataFrame
    recommendations_df = pd.DataFrame(recommendations)

    # Ensure Partner_id is a string
    recommendations_df["Partner_id"] = recommendations_df["Partner_id"].astype(str)

# Fix Recommended_Products: list of strings → string
    recommendations_df["Recommended_Products"] = recommendations_df["Recommended_Products"].apply(
    lambda prods: [str(p) for p in prods] if isinstance(prods, list) else []
    )
    recommendations_df["Recommended_Products"] = recommendations_df["Recommended_Products"].apply(str)


# Fix Similarity_Scores: stringified or list → list of floats → string
    
    recommendations_df["Similarity_Scores"] = recommendations_df["Similarity_Scores"].apply(str)

    #  Prepare test_data for evaluation
    test_data = test_data.reset_index()
    test_data["Partner_id"] = test_data["Partner_id"].astype(str)

    #  Save to S3
    save_file(test_data, "test_data.csv", is_lambda=True, bucket_name=os.environ.get("OUTPUT_BUCKET"))
    save_file(recommendations_df, "User_Based_Recommendations.csv", is_lambda=True, bucket_name=os.environ.get("OUTPUT_BUCKET"))

    return recommendations_df, test_data

