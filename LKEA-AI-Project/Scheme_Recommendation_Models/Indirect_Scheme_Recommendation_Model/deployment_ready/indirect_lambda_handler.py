import os
import logging
import boto3
import pandas as pd
from io import BytesIO

#-------------------import required files-----------------------
from Item_Based.Partner_product_recommendation_Evaluation import run_item_based_recommendation, run_evaluation
from Item_Based.User_Partner_Product_recom_Eval import run_user_based_recommendation
from Item_Based.linear_programing_partner_product__recommendation import run_lp_scheme_mapping
from Item_Based.optimized__partner_product_recommendations import run_simple_scheme_mapping
from Item_Based.final_scheme_recommendation_ import run_final_mapping


# ------------------- Logging Setup -------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ------------------- S3 Client Setup -------------------
s3_client = boto3.client('s3')

# ------------------- Helpers -------------------
def save_file(df, output_key, is_lambda, bucket_name):
    if is_lambda:
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        s3_client.put_object(Bucket=bucket_name, Key=output_key, Body=csv_buffer.getvalue())
        logger.info(f"Saved {output_key} to S3 bucket {bucket_name}")
    else:
        df.to_csv(output_key, index=False)
        logger.info(f"Saved {output_key} locally")

def load_file(input_key, is_lambda, bucket_name):
    if is_lambda:
        response = s3_client.get_object(Bucket=bucket_name, Key=input_key)
        df = pd.read_csv(response['Body'])
        logger.info(f"Loaded {input_key} from S3 bucket {bucket_name}")
    else:
        df = pd.read_csv(input_key)
        logger.info(f"Loaded {input_key} locally")
    return df

# ------------------- Main Lambda Handler -------------------
def lambda_handler(event=None, context=None):
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

    # -------------------- Step 0: Load Input File --------------------
    df = load_file(input_key, is_lambda, bucket_name)

    # -------------------- Step 1: Recommendation Generation --------------------
    if active_approach == 'item_based':
        logger.info("Running Item-Based Recommendation...")
        rec_df, test_df = run_item_based_recommendation(df, include_purchased)
    elif active_approach == 'user_based':
        logger.info("Running User-Based Recommendation...")
        rec_df, test_df = run_user_based_recommendation(df, include_purchased)
    else:
        raise ValueError(f"Invalid ACTIVE_APPROACH: {active_approach}")

    save_file(rec_df, recommendation_key, is_lambda, bucket_name)
    save_file(test_df, test_data_key, is_lambda, bucket_name)

    # -------------------- Step 2: Scheme Mapping --------------------
    if analysis_mode == 'lp':
        logger.info("Running LP-Based Scheme Mapping...")
        scheme_mapping_df = run_lp_scheme_mapping(df)
        save_file(scheme_mapping_df, mapping_key, is_lambda, bucket_name)
    elif analysis_mode == 'simple':
        logger.info("Running Simple Scheme Mapping...")
        scheme_mapping_df = run_simple_scheme_mapping(df)
        save_file(scheme_mapping_df, mapping_key, is_lambda, bucket_name)
    else:
        raise ValueError(f"Invalid ANALYSIS_MODE: {analysis_mode}")

    # -------------------- Step 3: Final Mapping --------------------
    logger.info("Running Final Mapping...")
    final_mapping_df = run_final_mapping(recommendation_key, mapping_key, is_lambda, bucket_name)
    save_file(final_mapping_df, final_output_key, is_lambda, bucket_name)

    # -------------------- Step 4: Evaluation --------------------
    logger.info("Running Evaluation...")
    evaluation_df = run_evaluation(test_data_key, recommendation_key, is_lambda, bucket_name)
    save_file(evaluation_df, evaluation_output_key, is_lambda, bucket_name)

    logger.info("===== Lambda Handler Execution Completed =====")
