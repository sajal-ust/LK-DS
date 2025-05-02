# Evaluation.py

import pandas as pd
import ast
import os

IS_LAMBDA = os.environ.get("IS_LAMBDA", "false").lower() == "true"
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "lk-scheme-recommendations")

def run_evaluation(recommendation_output_file):
    """Lambda-compatible Evaluation Code"""

    if IS_LAMBDA:
        import boto3
        from io import BytesIO
        s3 = boto3.client('s3')

        # Load from S3
        obj = s3.get_object(Bucket=OUTPUT_BUCKET, Key=recommendation_output_file)
        recommendations_df = pd.read_csv(BytesIO(obj['Body'].read()))

        obj_test = s3.get_object(Bucket=OUTPUT_BUCKET, Key="test_data.csv")
        test_df = pd.read_csv(BytesIO(obj_test['Body'].read()))
    else:
        recommendations_df = pd.read_csv(recommendation_output_file)
        test_df = pd.read_csv("test_data.csv")

    # Rename if needed
    recommendations_df = recommendations_df.rename(columns={"Partner_ID": "Partner_id"})

    # Identify product columns
    meta_cols = [
        'Partner_id', 'Stockist_Type', 'Scheme_Type', 'Sales_Value_Last_Period',
        'Sales_Quantity_Last_Period', 'MRP', 'Growth_Percentage', 'Discount_Applied',
        'Bulk_Purchase_Tendency', 'New_Stockist', 'Feedback_Score'
    ]
    product_cols = [col for col in test_df.columns if col not in meta_cols]

    # Create Purchased_Products list
    test_df[product_cols] = test_df[product_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    test_df['Purchased_Products'] = test_df[product_cols].apply(
        lambda row: [prod for prod, val in zip(product_cols, row) if val == 1], axis=1
    )
    test_df['has_purchase'] = test_df['Purchased_Products'].apply(lambda x: len(x) > 0)

    # Parse recommended product list
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

    # Evaluate Top-K
    results = []
    for k in [1, 2, 3]:
        precision_list, recall_list = [], []

        for _, row in df_all.iterrows():
            actual = set(row["Purchased_Products"])
            predicted = set(row["Recommended_Products"][:k])
            if not actual:
                continue

            tp = len(actual & predicted)
            precision = tp / k
            recall = tp / len(actual)

            precision_list.append(precision)
            recall_list.append(recall)

        avg_p = round(sum(precision_list) / len(precision_list), 4) if precision_list else 0
        avg_r = round(sum(recall_list) / len(recall_list), 4) if recall_list else 0
        f1 = round(2 * avg_p * avg_r / (avg_p + avg_r), 4) if (avg_p + avg_r) else 0

        results.append({
            "Top-K": k,
            "Avg Precision": avg_p,
            "Avg Recall": avg_r,
            "Avg F1 Score": f1
        })

    return pd.DataFrame(results)

