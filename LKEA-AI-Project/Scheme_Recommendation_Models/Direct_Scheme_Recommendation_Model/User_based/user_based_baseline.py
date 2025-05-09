import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def run_user_based_model_without_engagement():
    # Load dataset
    df = pd.read_csv(r"C:\Users\290938\OneDrive - UST\Desktop\LK-Git\LK-DS\LKEA-AI-Project\Scheme_Recommendation_Models\Direct_Scheme_Recommendation_Model\User_based\Augmented_Stockist_Data.csv")

    # Train-Test Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Partner_id"])
    train_df.to_csv("Train_Data.csv", index=False)
    test_df.to_csv("Test_Data.csv", index=False)

    # Pivot User-Scheme Matrix using frequency count
    scheme_counts = train_df.groupby(["Partner_id", "Scheme_Type"]).size().reset_index(name="Scheme_Count")
    user_scheme_matrix = scheme_counts.pivot_table(
        index="Partner_id", columns="Scheme_Type", values="Scheme_Count", fill_value=0
    )

    # Prepare sparse matrix
    user_scheme_sparse = csr_matrix(user_scheme_matrix.values)
    partner_id_lookup = list(user_scheme_matrix.index)

    # Fit Nearest Neighbors (Cosine Similarity)
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_scheme_sparse)

    # Recommendation Function
    def recommend_user_based(partner_id, top_n=3):
        if partner_id not in user_scheme_matrix.index:
            return None

        idx = partner_id_lookup.index(partner_id)
        distances, indices = knn_model.kneighbors(user_scheme_sparse[idx], n_neighbors=min(top_n + 1, len(user_scheme_matrix)))
        similarities = 1 - distances.flatten()
        neighbors = indices.flatten()

        filtered = [(i, sim) for i, sim in zip(neighbors, similarities) if i != idx]
        if not filtered:
            return None

        top_idx, sim_score = filtered[0]
        similar_user = partner_id_lookup[top_idx]
        sim_score = round(sim_score, 6)

        top_schemes = (
            train_df[train_df["Partner_id"] == similar_user]["Scheme_Type"]
            .value_counts().head(3).index.tolist()
        )
        while len(top_schemes) < 3:
            top_schemes.append("No Scheme")

        product = train_df[train_df["Partner_id"] == partner_id]["Product_id"].unique()[0]

        return [partner_id, product, sim_score, *top_schemes]

    # Generate Recommendations
    user_partners = test_df["Partner_id"].unique()
    user_recommendations = [recommend_user_based(pid) for pid in user_partners if recommend_user_based(pid)]

    # Save Output
    user_rec_df = pd.DataFrame(user_recommendations, columns=["Partner_id", "Product_id", "Similarity_Score", "Scheme_1", "Scheme_2", "Scheme_3"])
    user_rec_df.to_csv("user_based_recommendations_enhanced.csv", index=False)

    print("User-Based Recommendations saved.")

    # ----------- Evaluation -----------
    test_df = pd.read_csv("Test_Data.csv")
    rec_df = pd.read_csv("user_based_recommendations_enhanced.csv")

    # Group by Partner_id to get list of all availed schemes
    availed_df = (
        test_df.groupby("Partner_id")["Scheme_Type"]
        .apply(list)
        .reset_index()
        .rename(columns={"Scheme_Type": "Availed_Schemes"})
    )

    # Combine Scheme_1, Scheme_2, Scheme_3
    rec_df["Recommended_Schemes"] = rec_df[["Scheme_1", "Scheme_2", "Scheme_3"]].values.tolist()

    # Merge on Partner_id
    df_all = pd.merge(
        availed_df,
        rec_df[["Partner_id", "Recommended_Schemes"]],
        on="Partner_id",
        how="left"
    )

    # Sanitize columns
    df_all["Availed_Schemes"] = df_all["Availed_Schemes"].apply(lambda x: x if isinstance(x, list) else [])
    df_all["Recommended_Schemes"] = df_all["Recommended_Schemes"].apply(lambda x: x if isinstance(x, list) else [])

    # Top-K Evaluation
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
