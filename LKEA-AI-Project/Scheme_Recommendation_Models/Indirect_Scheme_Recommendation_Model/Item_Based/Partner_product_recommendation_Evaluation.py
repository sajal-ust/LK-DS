# Import necessary libraries 
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances

warnings.filterwarnings('ignore')

def run_item_based_recommendation(include_purchased=True, is_lambda=False):
    # Load dataset
    df = pd.read_csv("stockist_data.csv")

    # Define product columns
    product_cols = [
        "AIS(Air Insulated Switchgear)", "RMU(Ring Main Unit)", "PSS(Compact Sub-Stations)",
        "VCU(Vacuum Contactor Units)", "E-House", "VCB(Vacuum Circuit Breaker)",
        "ACB(Air Circuit Breaker)", "MCCB(Moduled Case Circuit Breaker)",
        "SDF(Switch Disconnectors)", "BBT(Busbar Trunking)", "Modular Switches"
    ]

    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print("Train and test data split successfully.")

    # Train product similarity
    df_products_train = train_df[product_cols].astype(bool)
    jaccard_product_sim = 1 - pairwise_distances(df_products_train.values.T, metric="jaccard")
    product_similarity_df = pd.DataFrame(jaccard_product_sim, index=product_cols, columns=product_cols)

    # Recommendation generation
    item_matrix = test_df.set_index("Partner_id")[product_cols].astype(bool)
    recommendations = []
    similarity_scores = []

    for index, row in test_df.iterrows():
        partner_id = row["Partner_id"]
        purchased_products = [p for p in product_cols if row[p] == 1]

        if not purchased_products:
            recommendations.append([])
            similarity_scores.append([])
            continue

        recommended_products = set()
        product_scores = []

        for product in purchased_products:
            top_products = product_similarity_df[product].sort_values(ascending=False)[1:4].index.tolist()
            recommended_products.update(top_products)
            scores = product_similarity_df.loc[product, top_products].values
            product_scores.extend(scores)

        recommended_products = list(recommended_products)

        if not include_purchased:
            already_purchased = item_matrix.loc[partner_id]
            recommended_products = [prod for prod in recommended_products if not already_purchased[prod]]

        final_recommendations = recommended_products[:3]
        final_scores = product_scores[:3]

        recommendations.append(final_recommendations)
        similarity_scores.append(final_scores)

        if index < 5:
            print(f"\nRecommendations for Partner {partner_id}:")
            print(f"Purchased Products: {purchased_products}")
            print(f"Recommended Products: {final_recommendations}")
            print(f"Similarity Scores: {final_scores}")

    test_df["Recommended_Products"] = recommendations
    test_df["Similarity_Scores"] = similarity_scores
    recommended_df = test_df[["Partner_id", "Recommended_Products", "Similarity_Scores"]]

    print("\nGenerated product recommendations successfully.")

    return recommended_df, test_df


# run_item_based_recommendation(include_purchased=True, is_lambda=False)



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
