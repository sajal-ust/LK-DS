import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def run_user_based_model(use_engagement_score=True):
    # Load dataset
    df = pd.read_csv("Augmented_Stockist_Dat.csv")

    # Compute Engagement Score if enabled
    if use_engagement_score:
        df["Engagement_Score"] = np.log1p(df["Sales_Value_Last_Period"]) * (
            df["Feedback_Score"] + df["Growth_Percentage"]
        )
        score_col = "Engagement_Score"
    else:
        df["Scheme_Indicator"] = 1
        score_col = "Scheme_Indicator"

    # Train-Test Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Partner_id"])
    train_df.to_csv("train_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)

    # Pivot matrix
    user_scheme_matrix = train_df.pivot_table(
        index="Partner_id", columns="Scheme_Type", values=score_col, aggfunc="mean", fill_value=0
    )

    # Prepare for similarity search
    user_scheme_sparse = csr_matrix(user_scheme_matrix.values)
    partner_id_lookup = list(user_scheme_matrix.index)

    # Fit KNN model
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_scheme_sparse)

    recommendations = []
    for partner_id in test_df["Partner_id"].unique():
        if partner_id not in user_scheme_matrix.index:
            continue

        idx = partner_id_lookup.index(partner_id)
        distances, indices = knn_model.kneighbors(user_scheme_sparse[idx], n_neighbors=min(4, len(user_scheme_matrix)))
        similarities = 1 - distances.flatten()
        neighbors = indices.flatten()

        filtered = [(i, sim) for i, sim in zip(neighbors, similarities) if i != idx]
        if not filtered:
            continue

        partner_products = test_df[test_df["Partner_id"] == partner_id]["Product_id"].unique()
        for product in partner_products:
            scheme_counts = pd.Series(dtype=int)
            for top_idx, sim_score in filtered:
                similar_user = partner_id_lookup[top_idx]
                similar_user_data = train_df[
                    (train_df["Partner_id"] == similar_user) &
                    (train_df["Product_id"] == product)
                ]
                schemes = similar_user_data["Scheme_Type"].value_counts()
                scheme_counts = scheme_counts.add(schemes, fill_value=0)

            top_schemes = scheme_counts.sort_values(ascending=False).head(3).index.tolist()
            while len(top_schemes) < 3:
                top_schemes.append("No Scheme")

            top_similarity = round(filtered[0][1], 6)
            recommendations.append([partner_id, product, top_similarity, *top_schemes])

    user_rec_df = pd.DataFrame(
        recommendations,
        columns=["Partner_id", "Product_id", "Similarity_Score", "Scheme_1", "Scheme_2", "Scheme_3"]
    )
    user_rec_df.to_csv("user_based_recommendations_enhanced.csv", index=False)
    print("User-Based Recommendations saved.")


