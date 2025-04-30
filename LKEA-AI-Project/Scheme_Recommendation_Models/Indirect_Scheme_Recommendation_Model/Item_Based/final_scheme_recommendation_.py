import pandas as pd
import ast
from Indirect_Scheme_Recommendation_Model.utils import load_file, save_file
import logging

logger = logging.getLogger()
def safe_eval(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) and val.startswith("[") else val
    except:
        return val

def run_final_mapping(recommendation_key, mapping_key, final_output_key, is_lambda=False, bucket_name=None):
    # Load the data using smart loader
    df_scheme_mapping = load_file(mapping_key, is_lambda, bucket_name)
    df_recommendations = load_file(recommendation_key, is_lambda, bucket_name)

    logger.info(f"df_recommendations.columns: {df_recommendations.columns.tolist()}")

    # Safely evaluate list-like strings into Python lists
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

    df_final_mapping = pd.DataFrame(results, columns=["Partner_id", "Product_id", "Similarity_Scores", "Scheme_1", "Scheme_2", "Scheme_3"])

    # Save final mapping using smart saver
    save_file(df_final_mapping, final_output_key, is_lambda, bucket_name)

    logger.info("Final Partner Product Schemes saved successfully!")

    return df_final_mapping  # Return the DataFrame

