import os
import logging
import boto3
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# ------------------ Lambda-compatible path builder ------------------
def get_local_path(filename):
    base = "/tmp" if os.getenv("IS_LAMBDA", "false").lower() == "true" else "."
    resolved_path = os.path.join(base, filename)
    logging.info(f"Resolved local path: {resolved_path}")
    return resolved_path

# ------------------ Logging Setup ------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# ------------------ I/O Helpers ------------------
s3_client = boto3.client("s3")

def load_file_from_s3(bucket, key):
    logger.info(f"Loading file from S3: {bucket}/{key}")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(response['Body'].read()))

def save_file_to_s3(df, bucket, key):
    logger.info(f"Saving result to S3: {bucket}/{key}")
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer)

def load_file_locally(path):
    logger.info(f"Loading file locally: {path}")
    return pd.read_csv(get_local_path(path))

def save_file_locally(df, path):
    final_path = get_local_path(path)
    logger.info(f"Saving result locally to resolved path: {final_path}")
    df.to_csv(final_path, index=False)

def save_evaluation_output(df, output_file, bucket_name):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    s3_client.put_object(Bucket=bucket_name, Key=output_file, Body=buffer.getvalue())

    #  This line ensures consistent output across environments (Lambda + Local/EC2)
    logger.info("\n==== Final Evaluation Table ====\n" + df.to_string(index=False))


# ------------------ Item-Based Recommendation ------------------
def run_item_based_recommendation(df, bucket_name):
    df["Engagement_Score"] = np.log1p(df["Sales_Value_Last_Period"]) * (df["Feedback_Score"] + df["Growth_Percentage"])
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    save_file_to_s3(test_df, bucket_name, test_data_key)
    partner_product_schemes = train_df.groupby(["Partner_id", "Product_id"])["Scheme_Type"].apply(list).reset_index()
    partner_product_schemes["Entity"] = partner_product_schemes["Partner_id"] + "_" + partner_product_schemes["Product_id"]
    mlb = MultiLabelBinarizer()
    scheme_matrix = pd.DataFrame(
        mlb.fit_transform(partner_product_schemes["Scheme_Type"]),
        index=partner_product_schemes["Entity"],
        columns=mlb.classes_
    ).T

    similarity_matrix = pd.DataFrame(index=scheme_matrix.index, columns=scheme_matrix.index, dtype=float)
    for i in range(len(scheme_matrix)):
        for j in range(len(scheme_matrix)):
            similarity_matrix.iloc[i, j] = jaccard_score(scheme_matrix.iloc[i], scheme_matrix.iloc[j]) if i != j else 1.0

    test_pairs = test_df[["Partner_id", "Product_id", "Scheme_Type"]].drop_duplicates()
    recommendations = []
    for _, row in test_pairs.iterrows():
        partner, product, current_scheme = row
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
    return pd.DataFrame(recommendations)

# ------------------ User-Based Recommendation -------------------
def run_user_based_recommendation(df, bucket_name):
    df["Engagement_Score"] = np.log1p(df["Sales_Value_Last_Period"]) * (df["Feedback_Score"] + df["Growth_Percentage"])
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Partner_id"])
    save_file_to_s3(test_df, bucket_name, test_data_key)
    matrix = train_df.pivot_table(index="Partner_id", columns="Scheme_Type", values="Engagement_Score", aggfunc="mean", fill_value=0)
    user_scheme_sparse = csr_matrix(matrix.values)
    partner_ids = list(matrix.index)
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_scheme_sparse)

    def recommend_for_user(pid, top_n=3):
        if pid not in matrix.index:
            return None
        idx = partner_ids.index(pid)
        distances, indices = knn.kneighbors(user_scheme_sparse[idx], n_neighbors=min(top_n + 1, len(matrix)))
        sims = 1 - distances.flatten()
        neighbors = indices.flatten()
        filtered = [(i, s) for i, s in zip(neighbors, sims) if i != idx]
        if not filtered:
            return None
        top_idx, sim = filtered[0]
        similar_user = partner_ids[top_idx]
        top_schemes = train_df[train_df["Partner_id"] == similar_user]["Scheme_Type"].value_counts().head(3).index.tolist()
        while len(top_schemes) < 3:
            top_schemes.append("No Scheme")
        product = train_df[train_df["Partner_id"] == pid]["Product_id"].unique()[0]
        return [pid, product, round(sim, 6), *top_schemes]

    recs = [recommend_for_user(pid) for pid in test_df["Partner_id"].unique() if recommend_for_user(pid)]
    return pd.DataFrame(recs, columns=["Partner_id", "Product_id", "Similarity_Score", "Scheme_1", "Scheme_2", "Scheme_3"])

# ------------------ Evaluation Logic ------------------
def evaluate_scheme_recommendations(test_df, rec_df):
    availed_df = (
        test_df.groupby("Partner_id")["Scheme_Type"]
        .apply(list)
        .reset_index()
        .rename(columns={"Scheme_Type": "Availed_Schemes"})
    )
    rec_df["Recommended_Schemes"] = rec_df[["Scheme_1", "Scheme_2", "Scheme_3"]].values.tolist()
    merged = pd.merge(availed_df, rec_df[["Partner_id", "Recommended_Schemes"]], on="Partner_id", how="left")
    merged["Availed_Schemes"] = merged["Availed_Schemes"].apply(lambda x: x if isinstance(x, list) else [])
    merged["Recommended_Schemes"] = merged["Recommended_Schemes"].apply(lambda x: x if isinstance(x, list) else [])

    results = []
    for k in [1, 2, 3]:
        precision_list = []
        recall_list = []
        for _, row in merged.iterrows():
            actual = set(row["Availed_Schemes"])
            recommended = row["Recommended_Schemes"][:k]
            if not actual:
                continue
            tp = len(set(recommended) & actual)
            precision = tp / k
            recall = tp / len(actual)
            precision_list.append(precision)
            recall_list.append(recall)
        ap = round(sum(precision_list) / len(precision_list), 4) if precision_list else 0
        ar = round(sum(recall_list) / len(recall_list), 4) if recall_list else 0
        f1 = round(2 * ap * ar / (ap + ar), 4) if ap + ar else 0
        results.append({
            "Top-K": k,
            "Avg Precision": ap,
            "Avg Recall": ar,
            "Avg F1 Score": f1
        })
    return pd.DataFrame(results)


# ------------------ Main Trigger ------------------
def main_handler(event=None, context=None):
    for k, v in event.items():
        os.environ[k] = str(v)

    # ------------------ ENV Config ------------------
    active_approach = os.getenv("ACTIVE_APPROACH", "user_based")
    is_lambda = os.getenv("IS_LAMBDA", "false").lower() == "true"
    bucket_name = os.getenv("BUCKET_NAME", "lk-scheme-recommendations")
    input_key = os.getenv("INPUT_KEY", "Augmented_Stockist_Dat.csv")
    output_key = os.getenv("OUTPUT_KEY", "test_data.csv")
    evaluation_output_key = os.getenv("EVALUATION_OUTPUT_KEY", "Scheme_Evaluation_Metrics.csv")
    test_data_key = os.getenv("TEST_DATA_KEY", "test_data.csv")
    
    logger.info(f"[ENV] IS_LAMBDA={is_lambda}, ACTIVE_APPROACH={active_approach}, BUCKET_NAME={bucket_name}, INPUT_KEY={input_key}")

    output_map = {
        "item_based": "Item_Based_Scheme_Recommendations.csv",
        "user_based": "User_Based_Scheme_Recommendations.csv"
    }

    try:
        # Step 1: Load Data
        df = load_file_from_s3(bucket_name, input_key) if is_lambda else load_file_locally(input_key)

        # Step 2: Run Recommendation
        if active_approach == "item_based":
            result_df = run_item_based_recommendation(df, bucket_name)
        elif active_approach == "user_based":
            result_df = run_user_based_recommendation(df, bucket_name)
        else:
            raise ValueError("ACTIVE_APPROACH must be 'item_based' or 'user_based'")

        # Step 3: Save Recommendation Output
        recommendation_output_key = output_map[active_approach]
        save_file_to_s3(result_df, bucket_name, recommendation_output_key)

        # Step 4: Load Evaluation Test Data
        test_df = load_file_from_s3(bucket_name, test_data_key) if is_lambda else load_file_locally(test_data_key)

        # Step 5: Evaluate 
        result_eval_df = evaluate_scheme_recommendations(test_df, result_df)
        save_evaluation_output(result_eval_df, evaluation_output_key, bucket_name)



        # Step 6: Success response
        logger.info(f"{active_approach} recommendation and evaluation completed successfully.")
        return {
            "statusCode": 200,
            "body": f"{active_approach} recommendation and evaluation completed successfully."
        }

    except Exception as e:
        logger.error(f"Error in Lambda execution: {str(e)}")
        return {
            "statusCode": 500,
            "body": str(e)
        }


        
          

#-----------------------------------------
if __name__ == "__main__":
    print(main_handler())
