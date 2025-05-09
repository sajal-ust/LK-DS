import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Augmented_Stockist_Data.csv")

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("Train_Data.csv", index=False)
test_df.to_csv("Test_Data.csv", index=False)

# Group schemes applied per (Partner, Product)
partner_product_schemes = train_df.groupby(["Partner_id", "Product_id"])["Scheme_Type"].apply(list).reset_index()
partner_product_schemes["Entity"] = partner_product_schemes["Partner_id"] + "_" + partner_product_schemes["Product_id"]

# Binary encode scheme presence per entity
mlb = MultiLabelBinarizer()
scheme_matrix = pd.DataFrame(
    mlb.fit_transform(partner_product_schemes["Scheme_Type"]),
    index=partner_product_schemes["Entity"],
    columns=mlb.classes_
).T

# Jaccard similarity between schemes
similarity_matrix = pd.DataFrame(index=scheme_matrix.index, columns=scheme_matrix.index, dtype=float)
for i in range(len(scheme_matrix)):
    for j in range(len(scheme_matrix)):
        if i != j:
            similarity_matrix.iloc[i, j] = jaccard_score(scheme_matrix.iloc[i], scheme_matrix.iloc[j])
        else:
            similarity_matrix.iloc[i, j] = 1.0

# Generate top-3 similar scheme recommendations per test (Partner, Product, Scheme)
test_pairs = test_df[["Partner_id", "Product_id", "Scheme_Type"]].drop_duplicates()
recommendations = []

for _, row in test_pairs.iterrows():
    partner, product, current_scheme = row["Partner_id"], row["Product_id"], row["Scheme_Type"]
    if current_scheme in similarity_matrix.index:
        similar_schemes = similarity_matrix.loc[current_scheme].drop(current_scheme).sort_values(ascending=False).head(3)
        sim_list = similar_schemes.index.tolist()
        recommendations.append({
            "Partner_id": partner,
            "Product_id": product,
            "Similarity_Score": round(similar_schemes.mean(), 6),
            "Scheme_1": sim_list[0] if len(sim_list) > 0 else "No Scheme",
            "Scheme_2": sim_list[1] if len(sim_list) > 1 else "No Scheme",
            "Scheme_3": sim_list[2] if len(sim_list) > 2 else "No Scheme"
        })
    else:
        print(f"Scheme '{current_scheme}' not found in training data.")


# Save output
recommendation_df = pd.DataFrame(recommendations)
recommendation_df.to_csv("Scheme_Recommendations.csv", index=False)

# Preview
print(recommendation_df.head())

"Evaluation code"
# Import required library
import pandas as pd

# Load the test data (long format — one row per availed scheme per partner)
test_df = pd.read_csv("Test_Data.csv")

# Load the recommendation data (top 3 recommended schemes per partner)
rec_df = pd.read_csv("Scheme_Recommendations.csv")

# Group by Partner_id to get list of all availed schemes
availed_df = (
    test_df.groupby("Partner_id")["Scheme_Type"]
    .apply(list)
    .reset_index()
    .rename(columns={"Scheme_Type": "Availed_Schemes"})
)

# Combine Scheme_1, Scheme_2, Scheme_3 into a single list column
rec_df["Recommended_Schemes"] = rec_df[["Scheme_1", "Scheme_2", "Scheme_3"]].values.tolist()

# Merge availed and recommended schemes using Partner_id
df_all = pd.merge(
    availed_df,
    rec_df[["Partner_id", "Recommended_Schemes"]],
    on="Partner_id",
    how="left"
)

# Ensure both lists are properly formatted
df_all["Availed_Schemes"] = df_all["Availed_Schemes"].apply(lambda x: x if isinstance(x, list) else [])
df_all["Recommended_Schemes"] = df_all["Recommended_Schemes"].apply(lambda x: x if isinstance(x, list) else [])

# Initialize variables
k_list = [1, 2, 3]
results = []

# Evaluate precision, recall, F1 for each Top-K level
for k in k_list:
    precision_list = []
    recall_list = []

    for _, row in df_all.iterrows():
        actual_set = set(row["Availed_Schemes"])
        recommended_k = row["Recommended_Schemes"][:k]  # Top-K recommendations

        if not actual_set:
            continue  # skip if no availed schemes

        # Count correct predictions in Top-K
        tp = sum([1 for scheme in recommended_k if scheme in actual_set])
        precision = tp / k
        recall = tp / len(actual_set)

        precision_list.append(precision)
        recall_list.append(recall)

   # Average the metrics across all partners
    avg_precision = round(sum(precision_list) / len(precision_list), 4) if precision_list else 0
    avg_recall = round(sum(recall_list) / len(recall_list), 4) if recall_list else 0
    f1 = round(2 * avg_precision * avg_recall / (avg_precision + avg_recall), 4) if (avg_precision + avg_recall) else 0

    results.append({
        "Top-K": k,
        "Avg Precision": avg_precision,
        "Avg Recall": avg_recall,
        "Avg F1 Score": f1
    })
        
# Print Top-K per-scheme evaluation metrics
print("==== Per-Scheme Evaluation (WITH Availed Schemes) ====")
for r in results:
    print(f"\nTop-{r['Top-K']}")
    print(f"  Avg Precision : {r['Avg Precision']}")
    print(f"  Avg Recall    : {r['Avg Recall']}")
    print(f"  Avg F1 Score  : {r['Avg F1 Score']}")



