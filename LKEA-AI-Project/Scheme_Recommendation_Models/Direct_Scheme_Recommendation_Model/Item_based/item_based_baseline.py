import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split

def run_item_based_model_without_engagement():
    """
    Runs item-based scheme recommendation using Jaccard similarity without engagement score.

    Returns:
        pd.DataFrame: Recommendation DataFrame
        pd.DataFrame: Test dataset for evaluation
    """
    # Load dataset
    df = pd.read_csv("Augmented_Stockist_Data.csv")

    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv("Train_Data.csv", index=False)
    test_df.to_csv("Test_Data.csv", index=False)

    # Group schemes applied per (Partner, Product)
    partner_product_schemes = train_df.groupby(["Partner_id", "Product_id"])["Scheme_Type"].apply(list).reset_index()
    partner_product_schemes["Entity"] = partner_product_schemes["Partner_id"] + "_" + partner_product_schemes["Product_id"]

    # Binary encode scheme presence per entity
    mlb = MultiLabelBinarizer()
    scheme_matrix = pd.DataFrame(
        mlb.fit_transform(partner_product_schemes["Scheme_Type"]),
        index=partner_product_schemes["Entity"],
        columns=mlb.classes_
    ).T

    # Jaccard similarity between schemes
    similarity_matrix = pd.DataFrame(index=scheme_matrix.index, columns=scheme_matrix.index, dtype=float)
    for i in range(len(scheme_matrix)):
        for j in range(len(scheme_matrix)):
            if i != j:
                similarity_matrix.iloc[i, j] = jaccard_score(scheme_matrix.iloc[i], scheme_matrix.iloc[j])
            else:
                similarity_matrix.iloc[i, j] = 1.0

    # Generate top-3 similar scheme recommendations per test (Partner, Product, Scheme)
    test_pairs = test_df[["Partner_id", "Product_id", "Scheme_Type"]].drop_duplicates()
    recommendations = []

    for _, row in test_pairs.iterrows():
        partner, product, current_scheme = row["Partner_id"], row["Product_id"], row["Scheme_Type"]
        if current_scheme in similarity_matrix.index:
            similar_schemes = similarity_matrix.loc[current_scheme].drop(current_scheme).sort_values(ascending=False).head(3)
            sim_list = similar_schemes.index.tolist()
            recommendations.append({
                "Partner_id": partner,
                "Product_id": product,
                "Similarity_Score": round(similar_schemes.mean(), 6),
                "Scheme_1": sim_list[0] if len(sim_list) > 0 else "No Scheme",
                "Scheme_2": sim_list[1] if len(sim_list) > 1 else "No Scheme",
                "Scheme_3": sim_list[2] if len(sim_list) > 2 else "No Scheme"
            })
        else:
            print(f"Scheme '{current_scheme}' not found in training data.")

    # Save and return
    recommendation_df = pd.DataFrame(recommendations)
    recommendation_df.to_csv("Scheme_Recommendations.csv", index=False)
    return recommendation_df, test_df
