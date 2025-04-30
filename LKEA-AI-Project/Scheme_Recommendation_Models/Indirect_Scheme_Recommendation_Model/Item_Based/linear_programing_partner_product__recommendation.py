
import pandas as pd
import numpy as np
from scipy.optimize import linprog

def run_lp_scheme_mapping(df=None, is_lambda=False):
    # ------------------ Step 1: Load Data ------------------
    if df is None:
       df = pd.read_csv("stockist_data.csv")

# ------------------ Step 2: Melt Products ------------------
    metadata_cols = [
    'Partner_id', 'Geography', 'Stockist_Type', 'Scheme_Type', 'Sales_Value_Last_Period',
    'Sales_Quantity_Last_Period', 'MRP', 'Growth_Percentage', 'Discount_Applied',
    'Bulk_Purchase_Tendency', 'New_Stockist', 'Feedback_Score'
    ]

    product_cols = [col for col in df.columns if col not in metadata_cols]

    df_melted = df.melt(
        id_vars=metadata_cols,
        value_vars=product_cols,
        var_name='Product_id',
        value_name='Has_Product'
    )

    df_melted = df_melted[df_melted['Has_Product'] == 1].drop(columns=['Has_Product'])

# ------------------ Step 3: Aggregate Sales ------------------
    product_schemes = df_melted.groupby(
    ["Partner_id", "Product_id", "Scheme_Type"]
    ).agg({
         "Sales_Value_Last_Period": "sum",
         "Sales_Quantity_Last_Period": "sum"
    }).reset_index()

    optimization_data = product_schemes[["Product_id", "Scheme_Type", "Sales_Value_Last_Period"]]

# ------------------ Step 4: LP Optimization ------------------
    def optimize_schemes(product_group):
        schemes = product_group["Scheme_Type"].unique()
        num_schemes = len(schemes)

        if num_schemes == 0:
            return [None, None, None]

        if num_schemes <= 3:
            return list(schemes) + [None] * (3 - num_schemes)

        scheme_sales = product_group.groupby("Scheme_Type")["Sales_Value_Last_Period"].sum()
        c = -scheme_sales.values
        bounds = [(0, 1)] * num_schemes

        res = linprog(c, bounds=bounds, method='highs', options={"disp": False})

        if res.success:
            x = res.x
            top_indices = np.argsort(x)[::-1][:3]
            top_schemes = [scheme_sales.index[i] for i in top_indices]
            return top_schemes + [None] * (3 - len(top_schemes))
        else:
           return [None, None, None]

# ------------------ Step 5: Apply Optimization ------------------
    optimized_list = (
        optimization_data
        .groupby("Product_id", group_keys=False)
        .apply(optimize_schemes)   # <-- FIXED (removed include_groups=False)
        .reset_index(drop=True)
    )

# convert to DataFrame directly
    optimized_df = pd.DataFrame(optimized_list.tolist(), columns=["Scheme_1", "Scheme_2", "Scheme_3"])

# Merge with Product ID
    product_ids = optimization_data["Product_id"].drop_duplicates().reset_index(drop=True)
    optimized_schemes = pd.concat([product_ids, optimized_df], axis=1)

# ------------------ Step 6: Final Merge ------------------
    partners_per_product = df_melted.groupby("Product_id")["Partner_id"].apply(list).reset_index()

    final_optimized_output = partners_per_product.merge(optimized_schemes, on="Product_id", how="left")
   # final_optimized_output.to_csv("Top_Optimized_Schemes_with_LP.csv", index=False)
    #print("Saved: Top_Optimized_Schemes_with_LP.csv")
    return final_optimized_output  # Return the DataFrame

