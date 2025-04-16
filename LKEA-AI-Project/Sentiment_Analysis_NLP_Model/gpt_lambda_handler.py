

import boto3
import pandas as pd
from io import BytesIO
import logging
import os  # To read environment variables for Lambda
import sys
import openai
import time
from dotenv import load_dotenv
from sklearn.metrics import classification_report

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Add handlers if they don't exist
if not logger.handlers:
    # Create a formatter that includes line numbers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

    # Add file handler
    file_handler = logging.FileHandler('/tmp/app.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# Ensure directories exist
os.makedirs('./input_data/', exist_ok=True)
os.makedirs('./output_folder/', exist_ok=True)
#os.makedirs('./simulation_output/', exist_ok=True)


# Try to import custom modules
try:
    import gpt_model
    logger.info("Successfully imported custom modules")
except ImportError as e:
    logger.error(f"Failed to import custom modules: {e}")
    sys.exit(1)

# Initialize S3 client
s3_client = boto3.client('s3')

# Flag to distinguish between Lambda and local execution
is_lambda = os.environ.get('IS_LAMBDA', 'false').lower() == 'true'
logger.info(f"Running in Lambda mode: {is_lambda}")

def load_file_from_s3(bucket_name, file_key):
    """Load file from S3 and return as a DataFrame"""
    try:
        logger.info(f"Loading file from S3: {bucket_name}/{file_key}")
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read()

        # Check file extension to determine how to read it
        if file_key.endswith('.xlsx') or file_key.endswith('.xls'):
            df = pd.read_excel(BytesIO(file_content))
        elif file_key.endswith('.csv'):
            df = pd.read_csv(BytesIO(file_content))
        else:
            raise ValueError(f"Unsupported file format for {file_key}")

        logger.info(f"File loaded successfully from S3: {bucket_name}/{file_key}")
        return df
    except Exception as e:
        logger.error(f"Error loading file from S3: {e}")
        raise

def save_file_to_s3(df, bucket_name, file_key):
    """Save DataFrame to S3 as CSV"""
    try:
        logger.info(f"Saving file to S3: {bucket_name}/{file_key}")
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        s3_client.put_object(Bucket=bucket_name, Body=csv_buffer, Key=file_key)
        logger.info(f"File saved successfully to S3: {bucket_name}/{file_key}")
    except Exception as e:
        logger.error(f"Error saving file to S3: {e}")
        raise

def load_file_locally(file_path):
    """Load file from local filesystem and return as a DataFrame"""
    try:
        logger.info(f"Loading file locally: {file_path}")

        # Check file extension to determine how to read it
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format for {file_path}")

        logger.info(f"File loaded successfully: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading file locally: {e}")
        raise

def save_file_locally(df, file_path):
    """Save DataFrame to local filesystem as CSV"""
    try:
        logger.info(f"Saving file locally: {file_path}")
        df.to_csv(file_path, index=False)
        logger.info(f"File saved successfully: {file_path}")
    except Exception as e:
        logger.error(f"Error saving file locally: {e}")
        raise
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
# Lambda Handler Function
def lambda_handler(event, context):
    # Initialize S3 client
    s3 = boto3.client('s3')

    # Check if the event has the required S3 bucket and file information
    if 'bucket_name' not in event or 'file_key' not in event:
        raise ValueError("Event must contain 'bucket_name' and 'file_key' parameters")

    # Get S3 bucket and file details from event
    bucket_name = event['roberta-data-text']
    file_key = event['file_key']

    # Download the file from S3
    local_file_path = 'channel_partner_feedback.csv'
    s3.download_file(bucket_name, file_key, local_file_path)

    # Load data from CSV
    df = pd.read_csv(local_file_path)  # Ensure it has 'Feedback_Text' and 'Sentiment' columns

    # Check if the necessary columns are in the DataFrame
    if "Feedback_Text" not in df.columns or "Sentiment" not in df.columns:
        raise ValueError("CSV file must contain 'Feedback_Text' and 'Sentiment' columns")

    # Analyze Sentiment for GPT (Apply rate-limiting with sleep to avoid overwhelming the API)
    df["gpt_prediction"] = df["Feedback_Text"].apply(
        lambda x: (time.sleep(1), get_gpt_sentiment(x))[1]
    )

    # Generate classification report
    print("\nGPT Model Performance:")
    report = classification_report(df["Sentiment"], df["gpt_prediction"])
    print(report)

    # Save results to a new CSV file locally
    output_file_path = 'gpt_output.csv'
    df.to_csv(output_file_path, index=False)

    # Optionally, upload the results to S3
    output_s3_key = 'results/gpt_output.csv'
    s3.upload_file(output_file_path, bucket_name, output_s3_key)

    # Return the classification report and S3 path
    return {
        'statusCode': 200,
        'body': {
            'message': 'Sentiment analysis completed successfully',
            'classification_report': report,
            's3_output_path': f"s3://{bucket_name}/{output_s3_key}"
        }
    }

if __name__ == "__main__":
    os.environ['IS_LAMBDA'] = 'false'
    logger.info("Starting execution of lambda_handler")
    result = lambda_handler({}, None)
    logger.info(f"Execution result: {result}")
