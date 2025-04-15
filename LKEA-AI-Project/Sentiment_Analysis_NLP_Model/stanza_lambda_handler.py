
import boto3
import pandas as pd
from io import BytesIO
import logging
import os  # To read environment variables for Lambda
import sys
import stanza

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Add handlers if they don't exist
if not logger.handlers:
    # Create a formatter that includes line numbers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

    # Add file handler
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# Ensure directories exist
os.makedirs('/tmp/input_data/', exist_ok=True)
os.makedirs('/tmp/output_folder/', exist_ok=True)
#os.makedirs('./simulation_output/', exist_ok=True)


# Try to import custom modules
try:
    import stanza_model
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
# Load Stanza NLP model once (outside the handler to reuse in container)
try:
    nlp = stanza.Pipeline(lang="en", processors="tokenize,sentiment", use_gpu=False)
except:
    stanza.download("en")
    nlp = stanza.Pipeline(lang="en", processors="tokenize,sentiment", use_gpu=False)

def get_stanza_sentiment(text, nlp):
    doc = nlp(text)
    sentiment_score = doc.sentences[0].sentiment  # 0 = Negative, 1 = Neutral, 2 = Positive
    return "Positive" if sentiment_score == 2 else "Negative"

def lambda_handler(event, context):
    input_bucket = "roberta-data-text"
    output_bucket = "roberta-data-text"

    input_file_key = "channel_partner_feedback.csv"  # File key in S3
    input_file_path = "/tmp/input_data/channel_partner_feedback.csv"  # Local path

    output_file_key = "output_folder/stanza_sentiment_results.csv"
    output_file_path = "/tmp/output_folder/stanza_sentiment_results.csv"

    try:
        # Handle both single string and list of strings
        input_texts = event.get("text") or event.get("texts")

        if isinstance(input_texts, str):
            input_texts = [input_texts]

        results = []
        for text in input_texts:
            prediction = get_stanza_sentiment(text, nlp)
            results.append({
                "input": text,
                "prediction": prediction
            })

        return {
            "statusCode": 200,
            "body": json.dumps({
                "results": results
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }
if __name__ == "__main__":
    os.environ['IS_LAMBDA'] = 'false'
    logger.info("Starting execution of lambda_handler")
    result = lambda_handler({}, None)
    logger.info(f"Execution result: {result}")