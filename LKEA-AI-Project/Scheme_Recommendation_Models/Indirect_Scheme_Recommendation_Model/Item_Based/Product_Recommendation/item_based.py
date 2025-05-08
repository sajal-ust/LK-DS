import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances

warnings.filterwarnings('ignore')

def run_item_based_model(input_df, include_purchased=True):
    """
    Generate item-based product recommendations using Jaccard similarity.

    Args:
        input_df (pd.DataFrame): Stockist input dataset.
        include_purchased (bool): Whether to include already purchased products in recommendations.

    Returns:
        pd.DataFrame: Partner_id, Recommended_Products, Similarity_Scores
    """
    # Define product columns (binary purchase indicators)
    product_cols = [
        "AIS(Air Insulated Switchgear)", "RMU(Ring Main Unit)", "PSS(Compact Sub-Stations)",
        "VCU(Vacuum Contactor Units)", "E-House", "VCB(Vacuum Circuit Breaker)",
        "ACB(Air Circuit Breaker)", "MCCB(Moduled Case Circuit Breaker)",
        "SDF(Switch Disconnectors)", "BBT(Busbar Trunking)", "Modular Switches"
    ]

    # Split data into train and test sets
    train_df, test_df = train_test_split(input_df, test_size=0.2, random_state=42)

    # Compute Jaccard similarity matrix between product columns
    df_products_train = train_df[product_cols].astype(bool)
    jaccard_product_sim = 1 - pairwise_distances(df_products_train.values.T, metric="jaccard")
    product_similarity_df = pd.DataFrame(jaccard_product_sim, index=product_cols, columns=product_cols)

    # Helper to get top 3 similar products
    def get_top3_products(product_name):
        if product_name not in product_similarity_df.index:
            return ["Product not found"]
        return list(product_similarity_df[product_name].sort_values(ascending=False)[1:4].index)

    # Prepare binary product matrix for exclusion logic
    item_matrix = test_df.set_index("Partner_id")[product_cols].astype(bool)

    recommendations, similarity_scores = [], []

    for _, row in test_df.iterrows():
        partner_id = row["Partner_id"]
        purchased_products = [p for p in product_cols if row[p] == 1]

        if not purchased_products:
            recommendations.append([])
            similarity_scores.append([])
            continue

        recommended_products = set()
        scores = []

        for product in purchased_products:
            top_prods = get_top3_products(product)
            recommended_products.update(top_prods)
            scores.extend(product_similarity_df.loc[product, top_prods].values)

        if not include_purchased:
            already_purchased = item_matrix.loc[partner_id]
            recommended_products = [prod for prod in recommended_products if not already_purchased[prod]]

        recommendations.append(list(recommended_products)[:3])
        similarity_scores.append(scores[:3])

    test_df["Recommended_Products"] = recommendations
    test_df["Similarity_Scores"] = similarity_scores

    return test_df[["Partner_id", "Recommended_Products", "Similarity_Scores"]]


# Optional: For testing locally
if __name__ == "__main__":
    df = pd.read_csv("stockist_data.csv")
    result = run_item_based_model(df, include_purchased=True)
    print(result.head())
    result.to_csv("Partner_Product_Recommendations.csv", index=False)
