import pandas as pd

# Load file assuming no header is present
df = pd.read_csv("Final_Recommendations.csv", header=None)

# Assign proper column names
df.columns = ["Partner_id", "Product_id", "Similarity_Score", "Scheme_1", "Scheme_2", "Scheme_3"]

# Save file again with correct headers
df.to_csv("Final_Recommendations.csv", index=False)

print(" Fixed header and saved Final_Recommendations.csv")
