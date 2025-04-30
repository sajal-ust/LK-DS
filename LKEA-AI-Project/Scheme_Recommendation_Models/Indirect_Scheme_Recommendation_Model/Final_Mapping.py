# Final_Mapping.py

import pandas as pd
import ast
import boto3
from io import BytesIO

# Setup S3 client
s3_client = boto3.client('s3')

def safe_eval(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) and val.startswith("[") else val
    except:
        return val

def run_final_mapping(recommendation_key, mapping_key, final_output_key, is_lambda=False, bucket_name=None):
    """Final Partner-Product-Scheme Mapping"""
    if is_lambda:
        # Load recommendation output from S3
        rec_obj = s3_client.get_object(Bucket=bucket_name, Key=recommendation_key)
        df_recommendations = pd.read_csv(BytesIO(rec_obj['Body'].read()))
        
        # Load scheme mapping from S3
        map_obj = s3_client.get_object(Bucket=bucket_name, Key=mapping_key)
        df_scheme_mapping = pd.read_csv(BytesIO(map_obj['Body'].read()))
    else:
        # Load locally
        df_recommendations = pd.read_csv(recommendation_key)
        df_scheme_mapping = pd.read_csv(mapping_key)

    # Safely evaluate list-like columns
    df_recommendations["Recommended_Products"] = df_recommendations["Recommended_Products"].apply(safe_eval)
    df_recommendations["Similarity_Scores"] = df_recommendations["Similarity_Scores"].apply(safe_eval)

    results = []

    for _, row in df_recommendations.iterrows():
        partner_id = row["Partner_id"]
        for product, score in zip(row["Recommended_Products"], row["Similarity_Scores"]):
            schemes = df_scheme_mapping[df_scheme_mapping["Product_id"] == product][["Scheme_1", "Scheme_2", "Scheme_3"]]
            if not schemes.empty:
                scheme_1, scheme_2, scheme_3 = schemes.iloc[0]
            else:
                scheme_1 = scheme_2 = scheme_3 = "Not Available"

            results.append([partner_id, product, score, scheme_1, scheme_2, scheme_3])

    df_final_mapping = pd.DataFrame(
        results,
        columns=["Partner_id", "Product_id", "Similarity_Scores", "Scheme_1", "Scheme_2", "Scheme_3"]
    )

    if is_lambda:
        # Save final output to S3
        buffer = BytesIO()
        df_final_mapping.to_csv(buffer, index=False)
        buffer.seek(0)
        s3_client.put_object(Bucket=bucket_name, Body=buffer, Key=final_output_key)
    else:
        # Save locally
        df_final_mapping.to_csv(final_output_key, index=False)

    print(" Final Partner Product Schemes saved successfully!")
    return df_final_mapping

