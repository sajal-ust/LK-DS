import os
import logging
import boto3
import pandas as pd
import numpy as np
import ast
from io import BytesIO, StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

# -------------------- Logging Setup --------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


product_cols = [
    "AIS(Air Insulated Switchgear)", "RMU(Ring Main Unit)", "PSS(Compact Sub-Stations)",
    "VCU(Vacuum Contactor Units)", "E-House", "VCB(Vacuum Circuit Breaker)",
    "ACB(Air Circuit Breaker)", "MCCB(Moduled Case Circuit Breaker)",
    "SDF(Switch Disconnectors)", "BBT(Busbar Trunking)", "Modular Switches"
]

s3_client = boto3.client("s3")

# -------------------- I/O Helpers --------------------
def load_file(key, is_lambda, bucket_name):
    if is_lambda:
        logger.info(f"Loading from S3: {bucket_name}/{key}")
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        return pd.read_csv(BytesIO(obj['Body'].read()))
    else:
        logger.info(f"Loading locally: {key}")
        return pd.read_csv(key)

def save_file(df, key, is_lambda, bucket_name):
    if is_lambda:
        buffer = StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        s3_client.put_object(Bucket=bucket_name, Key=key, Body=buffer.getvalue())
        logger.info(f"Saved to S3: {key}")
    else:
        df.to_csv(key, index=False)
        logger.info(f"Saved locally: {key}")

# -------------------- Item-Based Recommendation --------------------
def run_item_based(df, test_data_key, recommendation_key, is_lambda, bucket_name, include_purchased):
    logger.info("Running item-based recommendation...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    save_file(train_df, "train_data.csv", is_lambda, bucket_name)
    save_file(test_df, test_data_key, is_lambda, bucket_name)

    df_products_train = train_df[product_cols].astype(bool)
    jaccard_product_sim = 1 - pairwise_distances(df_products_train.values.T, metric="jaccard")
    product_similarity_df = pd.DataFrame(jaccard_product_sim, index=product_cols, columns=product_cols)

    def get_top3_products(product_name):
        if product_name not in product_similarity_df.index:
            return []
        return list(product_similarity_df[product_name].sort_values(ascending=False)[1:4].index)

    item_matrix = test_df.set_index("Partner_id")[product_cols].astype(bool)
    recommendations, similarity_scores = [], []

    for index, row in test_df.iterrows():
        partner_id = row["Partner_id"]
        purchased = [p for p in product_cols if row[p] == 1]

        if not purchased:
            recommendations.append([])
            similarity_scores.append([])
            continue

        recs, scores = set(), []

        for prod in purchased:
            top3 = get_top3_products(prod)
            recs.update(top3)
            scores.extend(product_similarity_df.loc[prod, top3].values)

        if not include_purchased:
            already_purchased = item_matrix.loc[partner_id]
            recs = [r for r in recs if not already_purchased[r]]

        final_recs = list(recs)[:3]
        final_scores = scores[:3]

        recommendations.append(final_recs)
        similarity_scores.append(final_scores)

    test_df["Recommended_Products"] = recommendations
    test_df["Similarity_Scores"] = similarity_scores
    output_df = test_df[["Partner_id", "Recommended_Products", "Similarity_Scores"]]
    save_file(output_df, recommendation_key, is_lambda, bucket_name)
    logger.info("Item-based recommendations generated successfully.")

# -------------------- User-Based Recommendation --------------------
def run_user_based(df, test_data_key, recommendation_key, is_lambda, bucket_name, include_purchased):
    logger.info("Running user-based recommendation...")
    user_product_matrix = df.set_index("Partner_id")[product_cols].astype(int)
    train_data, test_data = train_test_split(user_product_matrix, test_size=0.2, random_state=42)
    save_file(train_data.reset_index(), "train_data.csv", is_lambda, bucket_name)
    save_file(test_data.reset_index(), test_data_key, is_lambda, bucket_name)

    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(train_data)
    train_user_ids = train_data.index.tolist()
    recommendations = []

    for partner_id in test_data.index:
        partner_vector = test_data.loc[[partner_id]]
        distances, indices = model.kneighbors(partner_vector, n_neighbors=6)

        recommended = []
        for idx in indices.flatten()[1:]:
            neighbor_id = train_user_ids[idx]
            neighbor_prods = set(train_data.loc[neighbor_id][train_data.loc[neighbor_id] == 1].index)
            purchased = set(test_data.loc[partner_id][test_data.loc[partner_id] == 1].index)

            if include_purchased:
                recommended.extend(list(neighbor_prods))
            else:
                recommended.extend(list(neighbor_prods - purchased))

        final_recs = list(dict.fromkeys(recommended))[:3]
        recommendations.append({"Partner_id": partner_id, "Recommended_Products": final_recs})

    rec_df = pd.DataFrame(recommendations)
    save_file(rec_df, recommendation_key, is_lambda, bucket_name)
    logger.info("User-based recommendations generated successfully.")


# -------------------- Scheme Mapping Logic --------------------
import pandas as pd
import numpy as np
from collections import Counter
from scipy.optimize import linprog

def run_lp_scheme_mapping(df):
    metadata_cols = [
        'Partner_id', 'Geography', 'Stockist_Type', 'Scheme_Type', 'Sales_Value_Last_Period',
        'Sales_Quantity_Last_Period', 'MRP', 'Growth_Percentage', 'Discount_Applied',
        'Bulk_Purchase_Tendency', 'New_Stockist', 'Feedback_Score'
    ]
    product_cols = [col for col in df.columns if col not in metadata_cols]

    df_melted = df.melt(
        id_vars=metadata_cols,
        value_vars=product_cols,
        var_name='Product_id',
        value_name='Has_Product'
    )
    df_melted = df_melted[df_melted['Has_Product'] == 1].drop(columns=['Has_Product'])

    product_schemes = df_melted.groupby(
        ["Partner_id", "Product_id", "Scheme_Type"]
    ).agg({
        "Sales_Value_Last_Period": "sum"
    }).reset_index()

    optimization_data = product_schemes[["Product_id", "Scheme_Type", "Sales_Value_Last_Period"]]

    def optimize_schemes(product_group):
        schemes = product_group["Scheme_Type"].unique()
        num_schemes = len(schemes)
        if num_schemes == 0:
            return [None, None, None]
        if num_schemes <= 3:
            return list(schemes) + [None] * (3 - num_schemes)
        scheme_sales = product_group.groupby("Scheme_Type")["Sales_Value_Last_Period"].sum()
        c = -scheme_sales.values
        bounds = [(0, 1)] * num_schemes
        res = linprog(c, bounds=bounds, method='highs', options={"disp": False})
        if res.success:
            x = res.x
            top_indices = np.argsort(x)[::-1][:3]
            top_schemes = [scheme_sales.index[i] for i in top_indices]
            return top_schemes + [None] * (3 - len(top_schemes))
        else:
            return [None, None, None]

    optimized_list = (
        optimization_data
        .groupby("Product_id", group_keys=False)
        .apply(optimize_schemes)
        .reset_index(drop=True)
    )

    optimized_df = pd.DataFrame(optimized_list.tolist(), columns=["Scheme_1", "Scheme_2", "Scheme_3"])
    product_ids = optimization_data["Product_id"].drop_duplicates().reset_index(drop=True)
    optimized_schemes = pd.concat([product_ids, optimized_df], axis=1)

    partners_per_product = df_melted.groupby("Product_id")["Partner_id"].apply(list).reset_index()
    final_output = partners_per_product.merge(optimized_schemes, on="Product_id", how="left")
    final_output.to_csv("Top_Optimized_Schemes_with_LP.csv", index=False)
    print("Saved: Top_Optimized_Schemes_with_LP.csv")

def run_simple_scheme_mapping(df):
    product_columns = [
        'AIS(Air Insulated Switchgear)', 'RMU(Ring Main Unit)', 'PSS(Compact Sub-Stations)',
        'VCU(Vacuum Contactor Units)', 'E-House', 'VCB(Vacuum Circuit Breaker)',
        'ACB(Air Circuit Breaker)', 'MCCB(Moduled Case Circuit Breaker)',
        'SDF(Switch Disconnectors)', 'BBT(Busbar Trunking)', 'Modular Switches',
        'Starter', 'Controller', 'Solar Solutions', 'Pump Starter and Controller'
    ]
    existing_product_columns = [col for col in product_columns if col in df.columns]
    product_scheme_data = []

    for product in existing_product_columns:
        product_df = df[df[product] == 1]
        if product_df.empty:
            continue
        partner_ids = product_df['Partner_id'].dropna().astype(str).unique()
        scheme_data = product_df[['Scheme_Type', 'Sales_Quantity_Last_Period']].dropna()
        scheme_growth = Counter()
        for _, row in scheme_data.iterrows():
            schemes = row['Scheme_Type'].split(', ') if isinstance(row['Scheme_Type'], str) else []
            for scheme in schemes:
                scheme_growth[scheme] += row['Sales_Quantity_Last_Period']
        top_schemes = [s[0] for s in scheme_growth.most_common(3)]
        while len(top_schemes) < 3:
            top_schemes.append("No Scheme Available")
        product_scheme_data.append({
            'Product_id': product,
            'Partner_id': ', '.join(partner_ids),
            'Scheme_1': top_schemes[0],
            'Scheme_2': top_schemes[1],
            'Scheme_3': top_schemes[2]
        })

    final_df = pd.DataFrame(product_scheme_data)
    final_df.to_csv("Optimized_Product_Partner_Scheme_Mapping.csv", index=False)
    print("Saved: Optimized_Product_Partner_Scheme_Mapping.csv")

def run_scheme_mapping(df, mode="simple"):
    if mode == "lp":
        run_lp_scheme_mapping(df)
    elif mode == "simple":
        run_simple_scheme_mapping(df)
    else:
        raise ValueError("Invalid analysis mode. Use 'simple' or 'lp'.")



# -------------------- Final Mapping --------------------
def safe_eval(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) and val.startswith("[") else val
    except Exception as e:
        logger.warning(f"Eval failed for value: {val}, error: {e}")
        return val

def run_final_mapping(recommendation_key, mapping_key, final_output_key, is_lambda, bucket_name):
    logger.info("===== Starting Final Mapping Handler =====")
    df_scheme_mapping = load_file(mapping_key, is_lambda, bucket_name)
    df_recommendations = load_file(recommendation_key, is_lambda, bucket_name)

    df_scheme_mapping["Partner_id"] = df_scheme_mapping["Partner_id"].apply(safe_eval)
    df_recommendations["Recommended_Products"] = df_recommendations["Recommended_Products"].apply(safe_eval)
    df_recommendations["Similarity_Scores"] = df_recommendations["Similarity_Scores"].apply(safe_eval)

    results = []
    for _, row in df_recommendations.iterrows():
        partner_id = row["Partner_id"]
        for product, score in zip(row["Recommended_Products"], row["Similarity_Scores"]):
            schemes = df_scheme_mapping[df_scheme_mapping["Product_id"] == product][["Scheme_1", "Scheme_2", "Scheme_3"]]
            scheme_1, scheme_2, scheme_3 = schemes.iloc[0].fillna("Not Available").values if not schemes.empty else ("Not Available", "Not Available", "Not Available")
            results.append([partner_id, product, score, scheme_1, scheme_2, scheme_3])

    df_final = pd.DataFrame(results, columns=["Partner_id", "Product_id", "Similarity_Scores", "Scheme_1", "Scheme_2", "Scheme_3"])
    save_file(df_final, final_output_key, is_lambda, bucket_name)
    logger.info("===== Final Mapping Completed Successfully =====")

# -------------------- Evaluation --------------------
def run_evaluation(test_data_key, recommendation_key, evaluation_output_key, is_lambda, bucket_name):
    logger.info("===== Starting Evaluation =====")
    test_df = load_file(test_data_key, is_lambda, bucket_name)
    recommendations_df = load_file(recommendation_key, is_lambda, bucket_name)
    recommendations_df = recommendations_df.rename(columns={"Partner_ID": "Partner_id"})

    meta_cols = [
        'Partner_id', 'Stockist_Type', 'Scheme_Type', 'Sales_Value_Last_Period',
        'Sales_Quantity_Last_Period', 'MRP', 'Growth_Percentage', 'Discount_Applied',
        'Bulk_Purchase_Tendency', 'New_Stockist', 'Feedback_Score'
    ]
    product_cols_eval = [col for col in test_df.columns if col not in meta_cols]
    test_df[product_cols_eval] = test_df[product_cols_eval].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    test_df['Purchased_Products'] = test_df[product_cols_eval].apply(
        lambda row: [prod for prod, val in zip(product_cols_eval, row) if val == 1], axis=1
    )
    test_df['has_purchase'] = test_df['Purchased_Products'].apply(lambda x: len(x) > 0)

    recommendations_df["Recommended_Products"] = recommendations_df["Recommended_Products"].apply(ast.literal_eval)

    df_all = pd.merge(
        test_df[['Partner_id', 'Purchased_Products', 'has_purchase']],
        recommendations_df[['Partner_id', 'Recommended_Products']],
        on='Partner_id',
        how='left'
    )
    df_all['Recommended_Products'] = df_all['Recommended_Products'].apply(lambda x: x if isinstance(x, list) else [])

    results = []
    for k in [1, 2, 3]:
        precision_list, recall_list = [], []
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

        results.append({"Top-K": k, "Avg Precision": avg_precision, "Avg Recall": avg_recall, "Avg F1 Score": f1_k})

    eval_df = pd.DataFrame(results)
    save_file(eval_df, evaluation_output_key, is_lambda, bucket_name)
    logger.info("===== Evaluation Completed =====")
    for r in results:
        logger.info(f"Top-{r['Top-K']} | Precision: {r['Avg Precision']} | Recall: {r['Avg Recall']} | F1: {r['Avg F1 Score']}")

# -------------------- Lambda Handler --------------------
def lambda_handler(event=None, context=None):
    # Local testing config override
    os.environ["IS_LAMBDA"] = "false"
    os.environ["ACTIVE_APPROACH"] = "item_based"  # or "user_based"
    os.environ["ANALYSIS_MODE"] = "lp"  # or "simple"
    os.environ["BUCKET_NAME"] = "lk-scheme-recommendations"
    os.environ["INPUT_KEY"] = "stockist_data.csv"
    os.environ["TEST_DATA_KEY"] = "test_data.csv"
    os.environ["RECOMMENDATION_OUTPUT_KEY"] = "Partner_Product_Recommendations.csv"
    os.environ["MAPPING_INPUT_KEY"] = "Optimized_Product_Partner_Scheme_Mapping.csv"
    os.environ["FINAL_MAPPING_OUTPUT_KEY"] = "Final_Partner_Product_Schemes.csv"
    os.environ["EVALUATION_OUTPUT_KEY"] = "Evaluation_Metrics.csv"
    # -------------------- ENV Config --------------------
    active_approach = os.getenv("ACTIVE_APPROACH", "item_based")
    include_purchased = os.getenv("INCLUDE_PURCHASED", "true").lower() == "true"
    is_lambda = os.getenv("IS_LAMBDA", "false").lower() == "true"
    analysis_mode = os.getenv("ANALYSIS_MODE", "simple")
    bucket_name = os.getenv("BUCKET_NAME", "lk-scheme-recommendations")
    input_key = os.getenv("INPUT_KEY", "stockist_data.csv")
    test_data_key = os.getenv("TEST_DATA_KEY", "test_data.csv")
    recommendation_key = os.getenv("RECOMMENDATION_OUTPUT_KEY", "Partner_Product_Recommendations.csv")
    mapping_key = os.getenv("MAPPING_INPUT_KEY", "Optimized_Product_Partner_Scheme_Mapping.csv")
    final_output_key = os.getenv("FINAL_MAPPING_OUTPUT_KEY", "Final_Partner_Product_Schemes.csv")
    evaluation_output_key = os.getenv("EVALUATION_OUTPUT_KEY", "Scheme_Evaluation_Metrics.csv")
    logger.info("===== Lambda Handler Execution Started =====")
    df = load_file(input_key, is_lambda, bucket_name)

    if active_approach == "item_based":
        run_item_based(df, test_data_key, recommendation_key, is_lambda, bucket_name, include_purchased)
    elif active_approach == "user_based":
        run_user_based(df, test_data_key, recommendation_key, is_lambda, bucket_name, include_purchased)
    else:
        raise ValueError("Invalid ACTIVE_APPROACH environment variable.")

    logger.info(f"Running scheme mapping using: {analysis_mode} mode")
    run_scheme_mapping(df, mode=analysis_mode)
    run_final_mapping(recommendation_key, mapping_key, final_output_key, is_lambda, bucket_name)
    run_evaluation(test_data_key, recommendation_key, evaluation_output_key, is_lambda, bucket_name)
    logger.info("===== Lambda Handler Execution Completed =====")
    return {
        "statusCode": 200,
        "body": f"{active_approach} recommendation, scheme mapping and final mapping complete."
    }
