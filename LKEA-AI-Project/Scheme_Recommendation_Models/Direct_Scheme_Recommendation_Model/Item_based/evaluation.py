import pandas as pd

def run_evaluation(recommendation_file="Scheme_Recommendations.csv", test_data_file="test_data.csv"):
    """
    Evaluate Top-K scheme recommendations using precision, recall, and F1 score.

    Args:
        recommendation_file (str): Path to CSV file with Scheme_1, Scheme_2, Scheme_3.
        test_data_file (str): Path to test data CSV with actual availed Scheme_Type per Partner.

    Returns:
        list of dict: Evaluation metrics for Top-1, Top-2, and Top-3.
    """
    # Load test and recommendation data
    test_df = pd.read_csv(test_data_file)
    rec_df = pd.read_csv(recommendation_file)

    # Group actual schemes by partner
    availed_df = (
        test_df.groupby("Partner_id")["Scheme_Type"]
        .apply(list)
        .reset_index()
        .rename(columns={"Scheme_Type": "Availed_Schemes"})
    )

    # Combine recommended schemes into a list
    rec_df["Recommended_Schemes"] = rec_df[["Scheme_1", "Scheme_2", "Scheme_3"]].values.tolist()

    # Merge ground truth and predictions
    df_all = pd.merge(
        availed_df,
        rec_df[["Partner_id", "Recommended_Schemes"]],
        on="Partner_id",
        how="left"
    )

    # Sanitize lists
    df_all["Availed_Schemes"] = df_all["Availed_Schemes"].apply(lambda x: x if isinstance(x, list) else [])
    df_all["Recommended_Schemes"] = df_all["Recommended_Schemes"].apply(lambda x: x if isinstance(x, list) else [])

    # Evaluate Top-K metrics
    k_list = [1, 2, 3]
    results = []

    for k in k_list:
        precision_list = []
        recall_list = []

        for _, row in df_all.iterrows():
            actual_set = set(row["Availed_Schemes"])
            recommended_k = row["Recommended_Schemes"][:k]

            if not actual_set:
                continue

            tp = sum([1 for scheme in recommended_k if scheme in actual_set])
            precision = tp / k
            recall = tp / len(actual_set)

            precision_list.append(precision)
            recall_list.append(recall)

        avg_precision = round(sum(precision_list) / len(precision_list), 4) if precision_list else 0
        avg_recall = round(sum(recall_list) / len(recall_list), 4) if recall_list else 0
        f1 = round(2 * avg_precision * avg_recall / (avg_precision + avg_recall), 4) if (avg_precision + avg_recall) else 0

        results.append({
            "Top-K": k,
            "Avg Precision": avg_precision,
            "Avg Recall": avg_recall,
            "Avg F1 Score": f1
        })

    print("==== Per-Scheme Evaluation (WITH Availed Schemes) ====")
    for r in results:
        print(f"\nTop-{r['Top-K']}")
        print(f"  Avg Precision : {r['Avg Precision']}")
        print(f"  Avg Recall    : {r['Avg Recall']}")
        print(f"  Avg F1 Score  : {r['Avg F1 Score']}")

    return results
