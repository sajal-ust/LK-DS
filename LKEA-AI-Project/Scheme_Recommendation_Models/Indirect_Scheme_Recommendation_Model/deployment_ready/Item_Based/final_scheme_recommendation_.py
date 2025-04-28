import pandas as pd  # Used for working with tabular data (like Excel or CSV files)
import ast  # Used to safely convert strings that look like Python objects (like lists or dictionaries) into actual Python objects

def safe_eval(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) and val.startswith("[") else val
    except:
        return val

def run_final_mapping(recommendation_key, mapping_key, final_output_key, is_lambda=False, bucket_name=None):
    # Load the data
    df_scheme_mapping = pd.read_csv(mapping_key)
    df_recommendations = pd.read_csv(recommendation_key)

    print("df_recommendations.columns:", df_recommendations.columns.tolist())

    # Safe eval
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
    df_final_mapping.to_csv(final_output_key, index=False)

    print("Final Partner Product Schemes saved successfully!")

