import os
import sys
import logging
import pandas as pd
from io import BytesIO
import boto3

# --- Append User_based path ---
sys.path.append(os.path.abspath("../User_based"))

# --- Import model variant functions ---
from item_based_baseline import run_item_based_model_without_engagement
from item_based_engagement_score import run_item_based_model_with_engagement
from user_based_baseline import run_user_based_model_without_engagement
from user_based_engagement_score import run_user_based_model_with_engagement

from evaluation import run_evaluation

# --- Logging Setup ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- I/O Helpers ---
s3_client = boto3.client("s3")

def get_local_path(filename):
    base = "/tmp" if os.getenv("IS_LAMBDA", "false").lower() == "true" else "."
    return os.path.join(base, filename)

def load_file_from_s3(bucket, key):
    logger.info(f"Loading file from S3: {bucket}/{key}")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(response['Body'].read()))

def save_file_to_s3(df, bucket, key):
    logger.info(f"Saving file to S3: {bucket}/{key}")
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer)

def load_file_locally(filename):
    path = get_local_path(filename)
    logger.info(f"Loading file locally: {path}")
    return pd.read_csv(path)

def save_file_locally(df, filename):
    path = get_local_path(filename)
    logger.info(f"Saving file locally: {path}")
    df.to_csv(path, index=False)

def save_evaluation_output(df, output_file, bucket_name, is_lambda):
    if is_lambda:
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        s3_client.put_object(Bucket=bucket_name, Key=output_file, Body=buffer)
    else:
        save_file_locally(df, output_file)
    logger.info("\n==== Final Evaluation Table ====\n" + df.to_string(index=False))

# --- Main Handler ---
def main_handler(event=None, context=None):
    for k, v in event.items():
        os.environ[k] = str(v)

    model_variant = os.getenv("MODEL_VARIANT", "user_baseline")
    is_lambda = os.getenv("IS_LAMBDA", "false").lower() == "true"
    bucket_name = os.getenv("BUCKET_NAME", "lk-scheme-recommendations")
    input_key = os.getenv("INPUT_KEY", "Augmented_Stockist_Data.csv")
    output_key = os.getenv("OUTPUT_KEY", "Scheme_Recommendations.csv")
    evaluation_key = os.getenv("EVALUATION_OUTPUT_KEY", "Scheme_Evaluation_Metrics.csv")
    test_key = os.getenv("TEST_DATA_KEY", "test_data.csv")

    logger.info(f"[ENV] IS_LAMBDA={is_lambda}, MODEL_VARIANT={model_variant}")

    try:
        df = load_file_from_s3(bucket_name, input_key) if is_lambda else load_file_locally(input_key)

        if model_variant == "item_engagement":
            result_df, test_df = run_item_based_model_with_engagement()
        elif model_variant == "item_baseline":
            result_df, test_df = run_item_based_model_without_engagement()
        elif model_variant == "user_engagement":
            result_df, test_df = run_user_based_model_with_engagement()
        elif model_variant == "user_baseline":
            result_df, test_df = run_user_based_model_without_engagement()
        else:
            raise ValueError("MODEL_VARIANT must be one of: item_engagement, item_baseline, user_engagement, user_baseline")

        if is_lambda:
            save_file_to_s3(result_df, bucket_name, output_key)
            save_file_to_s3(test_df, bucket_name, test_key)
        else:
            save_file_locally(result_df, output_key)
            save_file_locally(test_df, test_key)

        result_eval_df = run_evaluation(recommendation_file=output_key, test_data_file=test_key)

        save_evaluation_output(result_eval_df, evaluation_key, bucket_name, is_lambda)

        logger.info(f"{model_variant} recommendation + evaluation completed.")
        return {
            "statusCode": 200,
            "body": f"{model_variant} recommendation + evaluation completed."
        }

    except Exception as e:
        logger.error(f"Execution Error: {str(e)}")
        return {"statusCode": 500, "body": str(e)}

# --- Local Entry Point ---
if __name__ == "__main__":
    env_vars = {
        "IS_LAMBDA": os.getenv("IS_LAMBDA", "false"),
        "MODEL_VARIANT": os.getenv("MODEL_VARIANT", "user_baseline"),
        "BUCKET_NAME": os.getenv("BUCKET_NAME", "lk-scheme-recommendations"),
        "INPUT_KEY": os.getenv("INPUT_KEY", "Augmented_Stockist_Data.csv"),
        "OUTPUT_KEY": os.getenv("OUTPUT_KEY", "Scheme_Recommendations.csv"),
        "EVALUATION_OUTPUT_KEY": os.getenv("EVALUATION_OUTPUT_KEY", "Scheme_Evaluation_Metrics.csv"),
        "TEST_DATA_KEY": os.getenv("TEST_DATA_KEY", "test_data.csv")
    }
    print(main_handler(env_vars))
