import pandas as pd
from collections import Counter
import ast
import os
from indirect_lambda_handler import save_file, load_file



SCHEME_MAPPING_SIMPLE_KEY = os.environ.get("SCHEME_MAPPING_SIMPLE_KEY", "Optimized_Product_Partner_Scheme_Mapping.csv")
IS_LAMBDA = os.environ.get("IS_LAMBDA", "false").lower() == "true"
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "lk-scheme-recommendations")

def run_simple_scheme_mapping(df):
    """Simple Scheme Mapping with Lambda S3 compatibility"""

    product_columns = [
        'AIS(Air Insulated Switchgear)', 'RMU(Ring Main Unit)', 'PSS(Compact Sub-Stations)',
        'VCU(Vacuum Contactor Units)', 'E-House', 'VCB(Vacuum Circuit Breaker)', 'ACB(Air Circuit Breaker)',
        'MCCB(Moduled Case Circuit Breaker)', 'SDF(Switch Disconnectors)', 'BBT(Busbar Trunking)',
        'Modular Switches', 'Starter', 'Controller', 'Solar Solutions', 'Pump Starter and Controller'
    ]
    existing_product_columns = [col for col in product_columns if col in df.columns]

    product_scheme_data = []

    for product in existing_product_columns:
        product_df = df[df[product] == 1]
        if product_df.empty:
            continue

        partner_ids = product_df['Partner_id'].dropna().astype(str).unique()
        scheme_data = product_df[['Scheme_Type', 'Sales_Quantity_Last_Period']].dropna()

        scheme_growth = Counter()
        for _, row in scheme_data.iterrows():
            schemes = row['Scheme_Type'].split(', ') if isinstance(row['Scheme_Type'], str) else []
            for scheme in schemes:
                scheme_growth[scheme] += row['Sales_Quantity_Last_Period']

        top_schemes = [s[0] for s in scheme_growth.most_common(3)]
        while len(top_schemes) < 3:
            top_schemes.append("No Scheme Available")

        product_scheme_data.append({
            'Product_id': product,
            'Partner_id': ', '.join(partner_ids),
            'Scheme_1': top_schemes[0],
            'Scheme_2': top_schemes[1],
            'Scheme_3': top_schemes[2]
        })

    final_df = pd.DataFrame(product_scheme_data)

    # Use Lambda-safe saving
    save_file(
    final_df,
    SCHEME_MAPPING_SIMPLE_KEY,
    is_lambda=IS_LAMBDA,
    bucket_name=OUTPUT_BUCKET,
    use_env_key=True
)


    print(f"{SCHEME_MAPPING_SIMPLE_KEY} saved successfully!")
    return final_df

