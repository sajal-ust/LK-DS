# -*- coding: utf-8 -*-
"""Final_Scheme_Recommendation .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16zUqiGR4hIS6aK-Y4zI0Ja-hBBpScSlH
"""

import pandas as pd # Used for working with tabular data (like Excel or CSV files)
import ast# Used to safely convert strings that look like Python objects (like lists or dictionaries) into actual Python objects

# Load the data from CSV files
df_scheme_mapping = pd.read_csv("Optimized_Product_Partner_Scheme_Mapping.csv")
df_recommendations = pd.read_csv("Partner_Product_Recommendations.csv")

def safe_eval(val):
    # Try to safely convert a string that looks like a list (e.g., "[1, 2, 3]") into a real Python list
    try:
    # Only attempt conversion if it's a string starting with a square bracket
        return ast.literal_eval(val) if isinstance(val, str) and val.startswith("[") else val
        # If conversion fails, just return the original value without crashing the code
    except:
        return val

# Convert string representations of lists into actual Python lists (if needed)
df_scheme_mapping["Partner_id"] = df_scheme_mapping["Partner_id"].apply(safe_eval)
df_recommendations["Recommended_Products"] = df_recommendations["Recommended_Products"].apply(safe_eval)
df_recommendations["Similarity_Scores"] = df_recommendations["Similarity_Scores"].apply(safe_eval)

# This will store the final mapped results for all partners
# Go through each partner's recommended products
results = []
for _, row in df_recommendations.iterrows():
    partner_id = row["Partner_id"]
    # Loop through each recommended product and its similarity score
    for product, score in zip(row["Recommended_Products"], row["Similarity_Scores"]):
        # Look up the top 3 schemes for the recommended product
        # Extract the schemes, or set defaults if no mapping is found
        schemes = df_scheme_mapping[df_scheme_mapping["Product_id"] == product][["Scheme_1", "Scheme_2", "Scheme_3"]]
        scheme_1, scheme_2, scheme_3 = schemes.iloc[0].fillna("Not Available").values if not schemes.empty else ("Not Available", "Not Available", "Not Available")
        # Store the result row: partner, product, score, and its suggested schemes
        results.append([partner_id, product, score, scheme_1, scheme_2, scheme_3])
# Convert results list into a new DataFrame
df_final_schemes = pd.DataFrame(results, columns=["Partner_id", "Product_id", "Similarity_Scores", "Scheme_1", "Scheme_2", "Scheme_3"])
df_final_schemes.to_csv("Final_Partner_Product_Schemes.csv", index=False)

print("Final Partner Product Schemes saved successfully!")



