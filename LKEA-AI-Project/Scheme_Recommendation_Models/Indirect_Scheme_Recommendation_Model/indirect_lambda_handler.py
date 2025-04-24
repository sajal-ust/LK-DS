import os
import logging
import boto3
import pandas as pd
import numpy as np
import ast
from io import BytesIO, StringIO
from collections import Counter
from scipy.optimize import linprog
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# -------------------- Logging Setup --------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# -------------------- ENV Config --------------------
active_approach = os.getenv("ACTIVE_APPROACH", "item_based")
analysis_mode = os.getenv("ANALYSIS_MODE", "simple")
is_lambda = os.getenv("IS_LAMBDA", "false").lower() == "true"

bucket_name = os.getenv("BUCKET_NAME", "lk-scheme-recommendations")
input_key = os.getenv("INPUT_KEY", "stockist_data.csv")
output_key = os.getenv("OUTPUT_KEY", "User_Based_Recommendations.csv")
test_key = os.getenv("TEST_DATA_KEY", "test_data.csv")
evaluation_key = os.getenv("EVALUATION_OUTPUT_KEY", "evaluation_metrics.csv")

s3_client = boto3.client("s3")

# -------------------- I/O Helpers --------------------
def load_file(path_or_key):
    if is_lambda:
        logger.info(f"Loading from S3: {path_or_key}")
        response = s3_client.get_object(Bucket=bucket_name, Key=path_or_key)
        return pd.read_csv(BytesIO(response['Body'].read()))
    else:
        logger.info(f"Loading locally: {path_or_key}")
        if not os.path.exists(path_or_key):
            logger.error(f"File not found: {path_or_key}")
            return pd.DataFrame()  # return empty DataFrame to avoid crash
        return pd.read_csv(path_or_key)

def save_file(df, path_or_key):
    if is_lambda:
        logger.info(f"Saving to S3: {path_or_key}")
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        s3_client.put_object(Bucket=bucket_name, Key=path_or_key, Body=buffer)
    else:
        logger.info(f"Saving locally: {path_or_key}")
        df.to_csv(path_or_key, index=False)

# -------------------- Product Columns --------------------
product_cols = [
    "AIS(Air Insulated Switchgear)", "RMU(Ring Main Unit)", "PSS(Compact Sub-Stations)",
    "VCU(Vacuum Contactor Units)", "E-House", "VCB(Vacuum Circuit Breaker)",
    "ACB(Air Circuit Breaker)", "MCCB(Moduled Case Circuit Breaker)", "SDF(Switch Disconnectors)",
    "BBT(Busbar Trunking)", "Modular Switches", "Starter", "Controller",
    "Solar Solutions", "Pump Starter and Controller"
]

# -------------------- Recommendation --------------------
def generate_user_recommendations(df):
    user_product_matrix = df.set_index("Partner_id")[product_cols].astype(int)
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_product_matrix)

    results = []
    for partner_id in df["Partner_id"]:
        if partner_id not in user_product_matrix.index:
            results.append([partner_id, [], []])
            continue
        distances, indices = knn.kneighbors(user_product_matrix.loc[[partner_id]], n_neighbors=6)
        similar_users = user_product_matrix.iloc[indices[0][1:]]
        similarity_scores = 1 - distances[0][1:]
        recommended_products = similar_users.T.dot(similarity_scores).sort_values(ascending=False)
        already_purchased = user_product_matrix.loc[partner_id]
        recommended_products = recommended_products[~already_purchased.astype(bool)]
        top_recs = list(recommended_products.head(3).index)
        top_scores = list(recommended_products.head(3).values)
        results.append([partner_id, top_recs, top_scores])

    out_df = pd.DataFrame(results, columns=["Partner_id", "Recommended_Products", "Similarity_Scores"])
    save_file(out_df, "User_Based_Recommendations.csv")

# -------------------- LP Scheme Mapping --------------------
def run_lp_scheme_mapping(df):
    metadata_cols = ['Partner_id', 'Geography', 'Stockist_Type', 'Scheme_Type', 'Sales_Value_Last_Period',
                     'Sales_Quantity_Last_Period', 'MRP', 'Growth_Percentage', 'Discount_Applied',
                     'Bulk_Purchase_Tendency', 'New_Stockist', 'Feedback_Score']
    product_cols_actual = [col for col in df.columns if col not in metadata_cols]
    melted = df.melt(id_vars=metadata_cols, value_vars=product_cols_actual,
                     var_name='Product_id', value_name='Has_Product')
    melted = melted[melted['Has_Product'] == 1].drop(columns=['Has_Product'])
    grouped = melted.groupby(["Partner_id", "Product_id", "Scheme_Type"]).agg({
        "Sales_Value_Last_Period": "sum",
        "Sales_Quantity_Last_Period": "sum"
    }).reset_index()
    opt_data = grouped[["Product_id", "Scheme_Type", "Sales_Value_Last_Period"]]

    def optimize(group):
        schemes = group["Scheme_Type"].unique()
        if len(schemes) <= 3:
            return list(schemes) + [None] * (3 - len(schemes))
        c = -group.groupby("Scheme_Type")["Sales_Value_Last_Period"].sum().values
        bounds = [(0, 1)] * len(schemes)
        linprog(c, bounds=bounds, method='highs')  # not using solution in this case
        return [None, None, None]

    result = opt_data.groupby("Product_id").apply(optimize).reset_index()
    result[["Scheme_1", "Scheme_2", "Scheme_3"]] = pd.DataFrame(result[0].tolist(), index=result.index)
    result = result.drop(columns=[0])
    partners = melted.groupby("Product_id")["Partner_id"].apply(list).reset_index()
    merged = partners.merge(result, on="Product_id", how="left")
    save_file(merged, "Optimized_Product_Partner_Scheme_Mapping.csv")

# -------------------- Simple Scheme Mapping --------------------
def run_simple_scheme_mapping(df):
    output = []
    for product in product_cols:
        product_df = df[df[product] == 1]
        if product_df.empty:
            continue
        partner_ids = product_df['Partner_id'].dropna().astype(str).unique()
        scheme_data = product_df[['Scheme_Type', 'Sales_Quantity_Last_Period']].dropna()
        counter = Counter()
        for _, row in scheme_data.iterrows():
            schemes = row['Scheme_Type'].split(', ') if isinstance(row['Scheme_Type'], str) else []
            for scheme in schemes:
                counter[scheme] += row['Sales_Quantity_Last_Period']
        top_schemes = [s[0] for s in counter.most_common(3)] + ["No Scheme Available"] * 3
        output.append({
            'Product_id': product,
            'Partner_id': list(partner_ids),
            'Scheme_1': top_schemes[0],
            'Scheme_2': top_schemes[1],
            'Scheme_3': top_schemes[2]
        })
    save_file(pd.DataFrame(output), "Optimized_Product_Partner_Scheme_Mapping.csv")

# -------------------- Final Mapping --------------------
def generate_final_mapping(output_reco_file):
    df_scheme = load_file("Optimized_Product_Partner_Scheme_Mapping.csv")
    df_reco = load_file(output_reco_file)

    def safe_eval(val):
        try:
            return ast.literal_eval(val) if isinstance(val, str) and val.startswith("[") else val
        except:
            return val

    df_scheme["partner_id"] = df_scheme["partner_id"].apply(safe_eval)
    df_reco["recommended_products"] = df_reco["recommended_products"].apply(safe_eval)
    df_reco["similarity_scores"] = df_reco["similarity_scores"].apply(safe_eval)

    results = []
    for _, row in df_reco.iterrows():
        for product, score in zip(row["recommended_products"], row["similarity_scores"]):
            schemes = df_scheme[df_scheme["product_id"] == product][["scheme_1", "scheme_2", "scheme_3"]]
            if not schemes.empty:
                scheme_1, scheme_2, scheme_3 = schemes.iloc[0].fillna("Not Available").values
            else:
                scheme_1 = scheme_2 = scheme_3 = "Not Available"
            results.append([row["partner_id"], product, score, scheme_1, scheme_2, scheme_3])

    df_final = pd.DataFrame(results, columns=["Partner_id", "Product_id", "Similarity_Scores", "Scheme_1", "Scheme_2", "Scheme_3"])
    save_file(df_final, "Final_Partner_Product_Schemes.csv")

# -------------------- Lambda Handler --------------------
def lambda_handler(event=None, context=None):
    try:
        df = load_file(input_key)
        if df.empty:
            raise FileNotFoundError(f"{input_key} not found")
        
        # Step 1: Split and Save Test Data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        save_file(test_df, test_key)
        
       # Step 2: Generate Recommendations on training data only
        if active_approach == "user_based":
            generate_user_recommendations(df)
            output_file = "User_Based_Recommendations.csv"
        elif active_approach == "item_based":
            generate_user_recommendations(df)  # Replace if item-based differs
            output_file = "User_Based_Recommendations.csv"
        else:
            raise ValueError("Invalid ACTIVE_APPROACH")

        if analysis_mode == "simple":
            run_simple_scheme_mapping(df)
        elif analysis_mode == "lp":
            run_lp_scheme_mapping(df)
        else:
            raise ValueError("Invalid ANALYSIS_MODE")

        generate_final_mapping(output_file)
        logger.info("Execution completed successfully.")
        return {"statusCode": 200, "body": "Execution completed successfully."}

    except Exception as e:
        logger.error(f"Error in Lambda execution: {str(e)}")
        return {"statusCode": 500, "body": str(e)}

# -------------------- Evaluation --------------------
import pandas as pd
from io import BytesIO, StringIO

# Load test data
if is_lambda:
    test_response = s3_client.get_object(Bucket=bucket_name, Key=test_key)
    test_df = pd.read_csv(BytesIO(test_response['Body'].read()))
    rec_response = s3_client.get_object(Bucket=bucket_name, Key=output_key)
    recommendations_df = pd.read_csv(BytesIO(rec_response['Body'].read()))
else:
    test_df = pd.read_csv(test_key)
    recommendations_df = pd.read_csv(output_key)


recommendations_df = recommendations_df.rename(columns={"Partner_ID": "Partner_id"})

meta_cols = [
    'Partner_id', 'Stockist_Type', 'Scheme_Type', 'Sales_Value_Last_Period',
    'Sales_Quantity_Last_Period', 'MRP', 'Growth_Percentage', 'Discount_Applied',
    'Bulk_Purchase_Tendency', 'New_Stockist', 'Feedback_Score'
]
product_cols = [col for col in test_df.columns if col not in meta_cols]

test_df[product_cols] = test_df[product_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
test_df['Purchased_Products'] = test_df[product_cols].apply(
    lambda row: [prod for prod, val in zip(product_cols, row) if val == 1], axis=1
)
test_df['has_purchase'] = test_df['Purchased_Products'].apply(lambda x: len(x) > 0)
recommendations_df["Recommended_Products"] = recommendations_df["Recommended_Products"].apply(eval)

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

result_df = pd.DataFrame(results)

if is_lambda:
    csv_buffer = StringIO()
    result_df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket_name, Key="evaluation_metrics.csv", Body=csv_buffer.getvalue())
    logger.info("Evaluation metrics uploaded to S3: evaluation_metrics.csv")
else:
    result_df.to_csv("evaluation_metrics.csv", index=False)
    print("Saved evaluation metrics to evaluation_metrics.csv")

print("===== Top-K Recommendation Evaluation (Corrected) =====")
for r in results:
    print(f"Top-{r['Top-K']}")
    print(f"  Avg Precision : {r['Avg Precision']}")
    print(f"  Avg Recall    : {r['Avg Recall']}")
    print(f"  Avg F1 Score  : {r['Avg F1 Score']}")

# -------------------- Local Debug --------------------
if __name__ == "__main__":
    os.environ["IS_LAMBDA"] = "false"
    os.environ["ACTIVE_APPROACH"] = "user_based"
    os.environ["ANALYSIS_MODE"] = "lp"
    os.environ["INPUT_KEY"] = "stockist_data.csv"
    print(lambda_handler())

