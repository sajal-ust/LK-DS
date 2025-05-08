# Import Required Libraries
import pandas as pd
import ast
import os
from io import BytesIO
import boto3

# Setup S3 client
s3_client = boto3.client('s3')

# Read Environment Variables
IS_LAMBDA = os.environ.get('IS_LAMBDA', 'false').lower() == 'true'
OUTPUT_BUCKET = os.environ.get('OUTPUT_BUCKET', 'lk-discount-model')

def run_final_mapping(recommendation_output_file, scheme_mapping_file):
    """Final Mapping: Attach Top Schemes to Recommended Products"""

    def load_file(file_name):
        """Helper to load CSV from S3 or Local"""
        if IS_LAMBDA:
            response = s3_client.get_object(Bucket=OUTPUT_BUCKET, Key=file_name)
            return pd.read_csv(BytesIO(response['Body'].read()))
        else:
            return pd.read_csv(file_name)

    def save_file(df, output_key):
        """Helper to save CSV to S3 or Local"""
        if IS_LAMBDA:
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            s3_client.put_object(Bucket=OUTPUT_BUCKET, Body=buffer, Key=output_key)
        else:
            df.to_csv(output_key, index=False)

    # Load recommendation output and scheme mapping
    df_scheme_mapping = load_file(scheme_mapping_file)
    df_recommendations = load_file(recommendation_output_file)

    # Safe literal eval for columns
    def safe_eval(val):
        try:
            return ast.literal_eval(val) if isinstance(val, str) and val.startswith("[") else val
        except:
            return val

    df_scheme_mapping["Partner_id"] = df_scheme_mapping["Partner_id"].apply(safe_eval)
    df_recommendations["Recommended_Products"] = df_recommendations["Recommended_Products"].apply(safe_eval)
    df_recommendations["Similarity_Scores"] = df_recommendations["Similarity_Scores"].apply(safe_eval)

    # Final Mapping Logic
    results = []

    for _, row in df_recommendations.iterrows():
        partner_id = row["Partner_id"]
        for product, score in zip(row["Recommended_Products"], row["Similarity_Scores"]):
            schemes = df_scheme_mapping[df_scheme_mapping["Product_id"] == product][["Scheme_1", "Scheme_2", "Scheme_3"]]
            if not schemes.empty:
                scheme_1, scheme_2, scheme_3 = schemes.iloc[0].fillna("Not Available").values
            else:
                scheme_1, scheme_2, scheme_3 = "Not Available", "Not Available", "Not Available"

            results.append([partner_id, product, score, scheme_1, scheme_2, scheme_3])

    # Save Final Output
    df_final_schemes = pd.DataFrame(results, columns=["Partner_id", "Product_id", "Similarity_Scores", "Scheme_1", "Scheme_2", "Scheme_3"])
    save_file(df_final_schemes, "Final_Partner_Product_Schemes.csv")

    print(" Final Partner-Product Scheme Mapping saved successfully!")

