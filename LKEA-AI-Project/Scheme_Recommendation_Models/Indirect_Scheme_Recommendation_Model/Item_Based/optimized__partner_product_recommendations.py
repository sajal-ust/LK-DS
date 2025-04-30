# Import Required Libraries
import pandas as pd  # For working with data in tables (DataFrames)
from collections import Counter  # For counting the frequency of items in a list or collection
import ast  # For safely converting strings that look like Python data structures back into Python objects

def run_simple_scheme_mapping(df):

    """Simple Scheme Mapping Code"""

    # Load dataset
    #file_path = "stockist_data.csv"
    #df = pd.read_csv(file_path)

    # Define relevant product columns
    product_columns = [
        'AIS(Air Insulated Switchgear)', 'RMU(Ring Main Unit)', 'PSS(Compact Sub-Stations)',
        'VCU(Vacuum Contactor Units)', 'E-House', 'VCB(Vacuum Circuit Breaker)', 'ACB(Air Circuit Breaker)',
        'MCCB(Moduled Case Circuit Breaker)', 'SDF(Switch Disconnectors)', 'BBT(Busbar Trunking)',
        'Modular Switches', 'Starter', 'Controller', 'Solar Solutions', 'Pump Starter and Controller'
    ]
    existing_product_columns = [col for col in product_columns if col in df.columns]

    # Initialize tracking variables
    product_scheme_data = []

    # Process each product separately
    for product in existing_product_columns:
        # Filter the DataFrame to get only those rows (stockists) who sold this product
        product_df = df[df[product] == 1]

        if product_df.empty:
            continue  # Skip this product if no stockists have sold it

        # Extract relevant columns
        partner_ids = product_df['Partner_id'].dropna().astype(str).unique()

        # Only keep the columns needed for scheme analysis (scheme type and sales quantity)
        scheme_data = product_df[['Scheme_Type', 'Sales_Quantity_Last_Period']].dropna()

        # Compute scheme impact (Weighted by Sales Quantity) for each product separately
        scheme_growth = Counter()

        for _, row in scheme_data.iterrows():
            schemes = row['Scheme_Type'].split(', ') if isinstance(row['Scheme_Type'], str) else []
            for scheme in schemes:
                scheme_growth[scheme] += row['Sales_Quantity_Last_Period']

        # Select top 3 schemes for this product
        top_schemes = [s[0] for s in scheme_growth.most_common(3)]
        while len(top_schemes) < 3:
            top_schemes.append("No Scheme Available")

        # Store the processed data
        product_scheme_data.append({
            'Product_id': product,
            'Partner_id': ', '.join(partner_ids),
            'Scheme_1': top_schemes[0],
            'Scheme_2': top_schemes[1],
            'Scheme_3': top_schemes[2]
        })

    # Convert results into a DataFrame
    final_df = pd.DataFrame(product_scheme_data)

    # Save intermediate results
    final_scheme_mapping_path = "Optimized_Product_Partner_Scheme_Mapping.csv"
    final_df.to_csv(final_scheme_mapping_path, index=False)
    print("Optimized_Product_Partner_Scheme_Mapping.csv saved successfully!")
    return final_df
