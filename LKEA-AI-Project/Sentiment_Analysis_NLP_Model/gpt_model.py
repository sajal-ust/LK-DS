import pandas as pd
import time
import os
import openai
from dotenv import load_dotenv
from sklearn.metrics import classification_report

# Load environment variables
load_dotenv()

# Get API key from .env
openai.api_key = os.getenv("OPENAI_API_KEY")
print("API Key:", openai.api_key)

# Check if API key is loaded
if not openai.api_key:
    raise ValueError("API key not found.")

# Function to get sentiment using GPT
def get_gpt_sentiment(text, retries=3):
    for _ in range(retries):
        try:
            # Log the feedback text being analyzed (only first 50 chars for brevity)
            print(f"Analyzing sentiment for: {text[:50]}...")  

            # Send request to OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                request_timeout=30,  # Increase timeout to 30 seconds
                messages=[ 
                    {"role": "system", "content": "Classify this text strictly as Positive or Negative only."},
                    {"role": "user", "content": text}
                ]
            )
            
            # Log the full API response for debugging
            print(f"Full API Response: {response}")

            # Extract sentiment from the response
            sentiment = response['choices'][0]['message']['content'].strip().lower()
            
            # Return "Positive" if positive sentiment is detected, else "Negative"
            return "Positive" if "positive" in sentiment else "Negative"
        
        except Exception as e:
            print(f"Error: {e}, Retrying...")
            time.sleep(3)  # Wait before retrying

    # If retries are exhausted, return default sentiment
    return "Negative"

# Load data from CSV
df = pd.read_csv("channel_partner_feedback.csv")  # Ensure it has 'Feedback_Text' and 'Sentiment' columns

# Check if the necessary columns are in the DataFrame
if "Feedback_Text" not in df.columns or "Sentiment" not in df.columns:
    raise ValueError("CSV file must contain 'Feedback_Text' and 'Sentiment' columns")

# Analyze Sentiment for GPT (Apply rate-limiting with sleep to avoid overwhelming the API)
df["gpt_prediction"] = df["Feedback_Text"].apply(
    lambda x: (time.sleep(1), get_gpt_sentiment(x))[1]
)

# Print classification report
print("\nGPT Model Performance:")
print(classification_report(df["Sentiment"], df["gpt_prediction"]))

# Save results to a new CSV file
df.to_csv("gpt_output.csv", index=False)

# Show the DataFrame with predictions
print(df)

# Optionally, upload the results to S3 (if you want to store the file in AWS S3)
import boto3

# Set up S3 client
s3 = boto3.client('s3')

# S3 bucket details (replace with your bucket name)
bucket_name = 'roberta-data-text'
s3_key = 'results/gpt_output.csv'  # Path in the S3 bucket

# Upload to S3
s3.upload_file('gpt_output.csv', bucket_name, s3_key)

print(f" Uploaded to s3://{bucket_name}/{s3_key}")
