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
    recommendations_df.to_csv("Partner_Product_Recommendations", index=False)
    print("Recommendations saved to 'Partner_Product_Recommendations.csv'")


#Evaluation Code

# Import Required Libraries
import pandas as pd

def run_evaluation(recommendation_output_file):
    """Evaluation Code"""

    # Load test data with one-hot encoded product columns
    test_df = pd.read_csv("test_data.csv")

    # Load recommendation output (Top-N recommendations per partner)
    recommendations_df = pd.read_csv(recommendation_output_file)

    # Fix column naming inconsistency if needed
    recommendations_df = recommendations_df.rename(columns={"Partner_ID": "Partner_id"})

    # Identify Product columns
    meta_cols = [
        'Partner_id', 'Stockist_Type', 'Scheme_Type', 'Sales_Value_Last_Period',
        'Sales_Quantity_Last_Period', 'MRP', 'Growth_Percentage', 'Discount_Applied',
        'Bulk_Purchase_Tendency', 'New_Stockist', 'Feedback_Score'
    ]
    product_cols = [col for col in test_df.columns if col not in meta_cols]

    # Convert Product Columns to Purchased List
    test_df[product_cols] = test_df[product_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    # Purchased product list per partner
    test_df['Purchased_Products'] = test_df[product_cols].apply(
        lambda row: [prod for prod, val in zip(product_cols, row) if val == 1], axis=1
    )

    # Remove non-buyers
    test_df['has_purchase'] = test_df['Purchased_Products'].apply(lambda x: len(x) > 0)

    # Ensure list parsing from string if needed
    recommendations_df["Recommended_Products"] = recommendations_df["Recommended_Products"].apply(eval)

    # Merge with test dataset
    df_all = pd.merge(
        test_df[['Partner_id', 'Purchased_Products', 'has_purchase']],
        recommendations_df[['Partner_id', 'Recommended_Products']],
        on='Partner_id',
        how='left'
    )

    # Fill any missing recommendations with empty list
    df_all['Recommended_Products'] = df_all['Recommended_Products'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    results = []

    for k in [1, 2, 3]:
        precision_list = []
        recall_list = []

        for _, row in df_all.iterrows():
            actual_set = set(row["Purchased_Products"])
            recommended_k = row["Recommended_Products"][:k]

            if not actual_set:
                continue  # skip if no purchases

            tp = len(set(recommended_k) & actual_set)
            precision = tp / k
            recall = tp / len(actual_set)

            precision_list.append(precision)
            recall_list.append(recall)

        avg_precision = round(sum(precision_list) / len(precision_list), 4) if precision_list else 0
        avg_recall = round(sum(recall_list) / len(recall_list), 4) if recall_list else 0
        f1_k = round(2 * avg_precision * avg_recall / (avg_precision + avg_recall), 4) if (avg_precision + avg_recall) else 0

        results.append({
            "Top-K": k,
            "Avg Precision": avg_precision,
            "Avg Recall": avg_recall,
            "Avg F1 Score": f1_k
        })

    # Display Results
    print("===== Top-K Recommendation Evaluation (Corrected) =====")
    for r in results:
        print(f"\nTop-{r['Top-K']}")
        print(f"  Avg Precision : {r['Avg Precision']}")
        print(f"  Avg Recall    : {r['Avg Recall']}")
        print(f"  Avg F1 Score  : {r['Avg F1 Score']}")

# run_evaluation(recommendation_output_file="Partner_Product_Recommendations.csv")
