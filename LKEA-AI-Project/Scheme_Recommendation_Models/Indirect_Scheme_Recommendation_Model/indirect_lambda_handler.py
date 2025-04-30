# ----------------- Import Required Libraries -----------------
import os
import logging
import boto3
import pandas as pd
import numpy as np
import ast
from io import BytesIO
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linprog
import warnings

warnings.filterwarnings('ignore')

# ----------------- Logging Setup -----------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# ----------------- Global Constants -----------------
PRODUCT_COLS = [
    "AIS(Air Insulated Switchgear)", "RMU(Ring Main Unit)", "PSS(Compact Sub-Stations)",
    "VCU(Vacuum Contactor Units)", "E-House", "VCB(Vacuum Circuit Breaker)",
    "ACB(Air Circuit Breaker)", "MCCB(Moduled Case Circuit Breaker)",
    "SDF(Switch Disconnectors)", "BBT(Busbar Trunking)", "Modular Switches"
]
INPUT_BUCKET = os.environ.get('INPUT_BUCKET', 'lk-discount-model')
OUTPUT_BUCKET = os.environ.get('OUTPUT_BUCKET', 'lk-discount-model')
INPUT_KEY = os.environ.get('INPUT_KEY', 'stockist_data.csv')
ACTIVE_APPROACH = os.environ.get('ACTIVE_APPROACH', 'item')  # 'item' or 'user'
SCHEME_MAPPING_APPROACH = os.environ.get('SCHEME_MAPPING_APPROACH', 'lp')  # 'lp' or 'simple'
IS_LAMBDA = os.environ.get('IS_LAMBDA', 'false').lower() == 'true'
INCLUDE_PURCHASED = os.environ.get('INCLUDE_PURCHASED', 'true').lower() == 'true'
TEST_KEY = os.environ.get("TEST_KEY", "test_data.csv")
PARTNER_PRODUCT_KEY = os.environ.get("PARTNER_PRODUCT_KEY", "Partner_Product_Recommendations.csv")
USER_PRODUCT_KEY = os.environ.get("USER_PRODUCT_KEY", "User_Based_Recommendations.csv")
SCHEME_MAPPING_LP_KEY = os.environ.get("SCHEME_MAPPING_LP_KEY", "Top_Optimized_Schemes_with_LP.csv")
SCHEME_MAPPING_SIMPLE_KEY = os.environ.get("SCHEME_MAPPING_SIMPLE_KEY", "Optimized_Product_Partner_Scheme_Mapping.csv")
FINAL_SCHEME_KEY = os.environ.get("FINAL_SCHEME_KEY", "Final_Partner_Product_Schemes.csv")
EVALUATION_KEY = os.environ.get("EVALUATION_KEY", "")  # will fallback to output_key if empty

s3_client = boto3.client('s3')

# ----------------- Helper Functions -----------------
def load_file():
    try:
        if IS_LAMBDA:
            logger.info(f"Loading from S3: {INPUT_BUCKET}/{INPUT_KEY}")
            response = s3_client.get_object(Bucket=INPUT_BUCKET, Key=INPUT_KEY)
            return pd.read_csv(BytesIO(response['Body'].read()))
        else:
            logger.info(f"Loading locally: {INPUT_KEY}")
            return pd.read_csv(INPUT_KEY)
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        raise

def save_file(df, output_key):
    try:
        if IS_LAMBDA:
            logger.info(f"Saving to S3: {OUTPUT_BUCKET}/{output_key}")
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            s3_client.put_object(Bucket=OUTPUT_BUCKET, Body=buffer, Key=output_key)
        else:
            logger.info(f"Saving locally: {output_key}")
            local_dir = os.path.dirname(output_key)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir, exist_ok=True)
            df.to_csv(output_key, index=False)
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise


def save_test_file(df, output_key):
    try:
        save_file(df, output_key)
    except Exception as e:
        logger.error(f"Failed to save test file: {e}")
        raise


# ----------------- Lambda Handler -----------------
def lambda_handler(event, context):
    logger.info("===== Lambda Handler Execution Started =====")

    try:
        df = load_file()

        # Run Recommendation
        if ACTIVE_APPROACH == 'item':
            from Item_Based.Partner_product_recommendation_Evaluation import run_item_based_recommendation
            result_df, test_df = run_item_based_recommendation(df, INCLUDE_PURCHASED)
            output_key = PARTNER_PRODUCT_KEY
        elif ACTIVE_APPROACH == 'user':
            from User_Based.User_based_recommendation_Evaluation import run_user_based_recommendation
            result_df, test_df = run_user_based_recommendation(df, INCLUDE_PURCHASED)
            output_key = USER_PRODUCT_KEY

        else:
            raise ValueError("Invalid ACTIVE_APPROACH value. Must be 'item' or 'user'.")

        save_file(result_df, output_key)
        save_test_file(test_df, TEST_KEY)


        # Scheme Mapping
       
        if SCHEME_MAPPING_APPROACH == 'lp':
           from Item_Based.linear_programing_partner_product__recommendation import run_lp_scheme_mapping
           scheme_df = run_lp_scheme_mapping()
           save_file(scheme_df, SCHEME_MAPPING_LP_KEY)
           scheme_file = SCHEME_MAPPING_LP_KEY
        elif SCHEME_MAPPING_APPROACH == 'simple':
           from Item_Based.optimized__partner_product_recommendations import run_simple_scheme_mapping
           run_simple_scheme_mapping()
           scheme_file = SCHEME_MAPPING_SIMPLE_KEY

        else:
            raise ValueError("Invalid SCHEME_MAPPING_APPROACH value. Must be 'lp' or 'simple'.")

        # Final Mapping
        
        from Final_Mapping import run_final_mapping
        run_final_mapping(output_key, scheme_file, FINAL_SCHEME_KEY, IS_LAMBDA, OUTPUT_BUCKET)


        # Evaluation
        
        from Evaluation import run_evaluation
        eval_input = EVALUATION_KEY if EVALUATION_KEY else output_key
        run_evaluation(eval_input)

        logger.info("===== All Steps Completed Successfully =====")

    except Exception as e:
        logger.error(f"Lambda Handler Failed: {e}")


if __name__ == "__main__":
    lambda_handler({}, {})

