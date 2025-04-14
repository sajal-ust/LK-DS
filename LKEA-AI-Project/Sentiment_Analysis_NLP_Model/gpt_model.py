import pandas as pd
import time
import os
import openai
from dotenv import load_dotenv
from sklearn.metrics import classification_report

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("API key not found.")
def get_gpt_sentiment(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            request_timeout=10,
            messages=[ 
                {"role": "system", "content": "Classify this text strictly as Positive or Negative only."},
                {"role": "user", "content": text}
            ]
        )
        sentiment = response['choices'][0]['message']['content'].strip().lower()
        return "Positive" if "positive" in sentiment else "Negative"
    except Exception as e:
        print(f"Error: {e}")
        return "Negative"

# Load Data
df = pd.read_csv("channel_partner_feedback.csv")  # Ensure it has 'Feedback_Text' and 'Sentiment' columns

# Check if the necessary columns are in the DataFrame
if "Feedback_Text" not in df.columns or "Sentiment" not in df.columns:
    raise ValueError("CSV file must contain 'Feedback_Text' and 'Sentiment' columns")

# Analyze Sentiment for GPT (Add rate-limiting with sleep)
df["gpt_prediction"] = df["Feedback_Text"].apply(
    lambda x: (time.sleep(1), get_gpt_sentiment(x))[1]
)

# Print classification report
print("\nGPT Model Performance:")
print(classification_report(df["Sentiment"], df["gpt_prediction"]))
df.to_csv("gpt_output.csv", index=False)
# Show the dataframe with predictions
print(df)

