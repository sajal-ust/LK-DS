import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def run_user_based_model(use_engagement_score=True):
    # Load dataset
    df = pd.read_csv("Augmented_Stockist_Data.csv")

    # Compute Engagement Score if enabled
    if use_engagement_score:
        df["Engagement_Score"] = np.log1p(df["Sales_Value_Last_Period"]) * (
            df["Feedback_Score"] + df["Growth_Percentage"]
        )
        score_col = "Engagement_Score"

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

            # Fixed: use sorted scheme counts to select top 3
            top_schemes = (
                scheme_counts.sort_values(ascending=False)
                .head(3)
                .index.tolist()
            )

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

    return user_rec_df, test_df

import pandas as pd

def run_evaluation(recommendation_file="Scheme_Recommendations.csv", test_data_file="test_data.csv"):
    """
    Evaluate Top-K scheme recommendations using precision, recall, and F1 score.

    Args:
        recommendation_file (str): Path to CSV file with Scheme_1, Scheme_2, Scheme_3.
        test_data_file (str): Path to test data CSV with actual availed Scheme_Type per Partner.

    Returns:
        list of dict: Evaluation metrics for Top-1, Top-2, and Top-3.
    """
    # Load test and recommendation data
    test_df = pd.read_csv(test_data_file)
    rec_df = pd.read_csv(recommendation_file)

    # Group actual schemes by partner
    availed_df = (
        test_df.groupby("Partner_id")["Scheme_Type"]
        .apply(list)
        .reset_index()
        .rename(columns={"Scheme_Type": "Availed_Schemes"})
    )

    # Combine recommended schemes into a list
    rec_df["Recommended_Schemes"] = rec_df[["Scheme_1", "Scheme_2", "Scheme_3"]].values.tolist()

    # Merge ground truth and predictions
    df_all = pd.merge(
        availed_df,
        rec_df[["Partner_id", "Recommended_Schemes"]],
        on="Partner_id",
        how="left"
    )

    # Sanitize lists
    df_all["Availed_Schemes"] = df_all["Availed_Schemes"].apply(lambda x: x if isinstance(x, list) else [])
    df_all["Recommended_Schemes"] = df_all["Recommended_Schemes"].apply(lambda x: x if isinstance(x, list) else [])

    # Evaluate Top-K metrics
    k_list = [1, 2, 3]
    results = []

    for k in k_list:
        precision_list = []
        recall_list = []

        for _, row in df_all.iterrows():
            actual_set = set(row["Availed_Schemes"])
            recommended_k = row["Recommended_Schemes"][:k]

            if not actual_set:
                continue

            tp = sum([1 for scheme in recommended_k if scheme in actual_set])
            precision = tp / k
            recall = tp / len(actual_set)

            precision_list.append(precision)
            recall_list.append(recall)

        avg_precision = round(sum(precision_list) / len(precision_list), 4) if precision_list else 0
        avg_recall = round(sum(recall_list) / len(recall_list), 4) if recall_list else 0
        f1 = round(2 * avg_precision * avg_recall / (avg_precision + avg_recall), 4) if (avg_precision + avg_recall) else 0

        results.append({
            "Top-K": k,
            "Avg Precision": avg_precision,
            "Avg Recall": avg_recall,
            "Avg F1 Score": f1
        })

    print("==== Per-Scheme Evaluation (WITH Availed Schemes) ====")
    for r in results:
        print(f"\nTop-{r['Top-K']}")
        print(f"  Avg Precision : {r['Avg Precision']}")
        print(f"  Avg Recall    : {r['Avg Recall']}")
        print(f"  Avg F1 Score  : {r['Avg F1 Score']}")

    return results
