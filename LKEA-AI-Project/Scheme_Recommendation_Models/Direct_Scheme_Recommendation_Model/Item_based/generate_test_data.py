import pandas as pd

# Correct column names: Partner_id and Scheme_Type
data = {
    "Partner_id": ["P1001", "P1002", "P1003", "P1004"],
    "Scheme_Type": ["Cashback", "Loyalty Points", "Seasonal Offer", "Volume Discount"]
}

df = pd.DataFrame(data)
df.to_csv("test_data.csv", index=False)
print(" test_data.csv created successfully!")
