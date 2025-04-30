# Evaluation.py

import pandas as pd
import ast

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

    test_df['Purchased_Products'] = test_df[product_cols].apply(
        lambda row: [prod for prod, val in zip(product_cols, row) if val == 1], axis=1
    )

    test_df['has_purchase'] = test_df['Purchased_Products'].apply(lambda x: len(x) > 0)

    # Safely parse list if needed
    recommendations_df["Recommended_Products"] = recommendations_df["Recommended_Products"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    df_all = pd.merge(
        test_df[['Partner_id', 'Purchased_Products', 'has_purchase']],
        recommendations_df[['Partner_id', 'Recommended_Products']],
        on='Partner_id',
        how='left'
    )

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
                continue

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

    print("===== Top-K Recommendation Evaluation (Corrected) =====")
    for r in results:
        print(f"\nTop-{r['Top-K']}")
        print(f"  Avg Precision : {r['Avg Precision']}")
        print(f"  Avg Recall    : {r['Avg Recall']}")
        print(f"  Avg F1 Score  : {r['Avg F1 Score']}")
