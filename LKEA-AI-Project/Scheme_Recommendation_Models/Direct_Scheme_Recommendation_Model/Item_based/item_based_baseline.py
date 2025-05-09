import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split

# --- Load dataset ---
df = pd.read_csv(r"Augmented_Stockist_Data.csv")

# --- Compute Engagement Score ---
df["Engagement_Score"] = np.log1p(df["Sales_Value_Last_Period"]) * (
    df["Feedback_Score"] + df["Growth_Percentage"]
)

# --- Train-test split ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv("Train_Data.csv", index=False)
test_df.to_csv("Test_Data.csv", index=False)
# --- Partner × Scheme matrix ---
item_scheme_matrix = train_df.pivot_table(
    index="Partner_id",
    columns="Scheme_Type",
    values="Engagement_Score",
    aggfunc="mean",
    fill_value=0
)

# --- Compute dynamic threshold (mean of all non-zero scores) ---
non_zero_scores = item_scheme_matrix[item_scheme_matrix > 0].stack()
threshold = non_zero_scores.mean()

# --- Binarize using dynamic threshold ---
binary_scheme_matrix = (item_scheme_matrix >= threshold).astype(int)

# --- Transpose to get Scheme × Partner matrix ---
scheme_matrix = binary_scheme_matrix.T

# --- Jaccard similarity between schemes ---
similarity_matrix = pd.DataFrame(index=scheme_matrix.index, columns=scheme_matrix.index, dtype=float)

for i in range(len(scheme_matrix)):
    for j in range(len(scheme_matrix)):
        if i != j:
            similarity_matrix.iloc[i, j] = jaccard_score(
                scheme_matrix.iloc[i].values, scheme_matrix.iloc[j].values
            )
        else:
            similarity_matrix.iloc[i, j] = 1.0

# --- Generate top-3 scheme recommendations per test pair ---
recommendations = []
test_pairs = test_df[["Partner_id", "Product_id", "Scheme_Type"]].drop_duplicates()

for _, row in test_pairs.iterrows():
    partner = row["Partner_id"]
    product = row["Product_id"]
    current_scheme = row["Scheme_Type"]

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

# --- Save recommendations ---
recommendation_df = pd.DataFrame(recommendations)
recommendation_df.to_csv("Scheme_Recommendations.csv", index=False)

# --- Preview output ---
print(recommendation_df.head())



