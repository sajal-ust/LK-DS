import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

PRODUCT_COLUMNS = [
    "AIS(Air Insulated Switchgear)", "RMU(Ring Main Unit)", "PSS(Compact Sub-Stations)",
    "VCU(Vacuum Contactor Units)", "E-House", "VCB(Vacuum Circuit Breaker)",
    "ACB(Air Circuit Breaker)", "MCCB(Moduled Case Circuit Breaker)",
    "SDF(Switch Disconnectors)", "BBT(Busbar Trunking)", "Modular Switches"
]

def run_user_based_model(df, include_purchased=True):
    """
    Generate user-based product recommendations using KNN with cosine similarity.

    Args:
        df (pd.DataFrame): Input DataFrame with binary product indicators.
        include_purchased (bool): Whether to include already purchased products in recommendations.

    Returns:
        pd.DataFrame: Partner_id, Recommended_Products, Similarity_Scores
    """
    user_product_matrix = df.set_index("Partner_id")[PRODUCT_COLUMNS].astype(int)
    train_data, test_data = train_test_split(user_product_matrix, test_size=0.2, random_state=42)

    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(train_data)
    train_user_ids = train_data.index.tolist()

    rows = []

    for partner_id in test_data.index:
        partner_vector = test_data.loc[[partner_id]]
        distances, indices = model.kneighbors(partner_vector, n_neighbors=6)

        recommended_products = []
        similarity_list = []

        for idx, dist in zip(indices.flatten(), distances.flatten()):
            neighbor_id = train_user_ids[idx]

            if str(neighbor_id) == str(partner_id):
                continue

            similarity = 1 - dist
            similarity_list.append(similarity)

            neighbor_products = set(train_data.loc[neighbor_id][train_data.loc[neighbor_id] == 1].index)

            if include_purchased:
                recommended_products.extend(list(neighbor_products))
            else:
                purchased_products = set(test_data.loc[partner_id][test_data.loc[partner_id] == 1].index)
                recommended_products.extend(list(neighbor_products - purchased_products))

        # Remove duplicates and take top 3
        top_products = list(dict.fromkeys(recommended_products))[:3]
        avg_similarity = round(np.mean(similarity_list[1:]), 6) if len(similarity_list) > 1 else 0

        rows.append({
            "Partner_id": str(partner_id),
            "Recommended_Products": top_products,
            "Similarity_Scores": [avg_similarity] * len(top_products)
        })

    return pd.DataFrame(rows)

# Optional: For local testing
if __name__ == "__main__":
    df = pd.read_csv("stockist_data.csv")
    result = run_user_based_model(df, include_purchased=True)
    print(result.head())
    result.to_csv("User_Based_Recommendations.csv", index=False)

