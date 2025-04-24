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
    logger.addHandler(file_handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Flag to distinguish between Lambda and local execution
is_lambda = os.environ.get('IS_LAMBDA', 'false').lower() == 'true'
logger.info(f"Running in Lambda mode: {is_lambda}")

BASE_DIR = '/tmp' if is_lambda else '.'
# Ensure directories exist
os.makedirs(f'{BASE_DIR}/input_data', exist_ok=True)
os.makedirs(f'{BASE_DIR}/output_folder', exist_ok=True)

# Try to import custom modules
try:

    logger.info("Successfully imported custom modules")
except ImportError as e:
    logger.error(f"Failed to import custom modules: {e}")
    sys.exit(1)

# Initialize S3 client
s3_client = boto3.client('s3')



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

def save_file_locally(df, file_path="./gpt_lambda_output.csv"):
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


# Function to get sentiment using GPT
def get_gpt_batch_sentiment_with_score(texts, batch_size=50, timeout=20):
    """Process a batch of texts and return their sentiments and sentiment scores."""
    # Construct the prompt by numbering each feedback
    batched_prompt = "\n\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
    message_content = (
        "Classify each of the following texts as Positive or Negative and provide a sentiment score between 0 and 1 (higher value means more positive). "
        "Return the results in the format '1: Positive, 0.85' or '1: Negative, 0.15' for each line.\n\n"
        f"{batched_prompt}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            request_timeout=timeout,
            messages=[
                {"role": "system", "content": "You are a sentiment classifier, classify sentiment in Positive or Negative and provide a score between 0 and 1."},
                {"role": "user", "content": message_content}
            ]
        )
        # Get the raw text response
        output = response['choices'][0]['message']['content'].strip()
        # Split by lines and extract the sentiment and score for each item
        sentiments = []
        scores = []
        for line in output.splitlines():
            try:
                sentiment, score = line.split(",", 1)
                sentiment = sentiment.split(":")[1].strip()
                score = float(score.strip())
                sentiments.append(sentiment)
                scores.append(score)
            except ValueError:
                sentiments.append("Negative")  # Default if parsing fails
                scores.append(0.0)  # Default score
        return sentiments, scores
    except Exception as e:
        print(f"Error in batch processing: {e}")
        # Return default "Negative" with a score of 0 for each input in the batch
        return ["Negative"] * len(texts), [0.0] * len(texts)
#
def save_results_to_csv_wrapper(merged_df, output_s3_key, bucket_name=None, is_lambda):
    """
    Save results to local or S3 based on execution environment.
    """
    output_dir = '/tmp' if is_lambda else './output_folder'
    os.makedirs(output_dir, exist_ok=True)

    save_file_to_s3(merged_df, bucket_name, output_s3_key)

    try:
        if is_lambda:
            logger.info("Lambda environment detected — uploading to S3.")
            if bucket_name:
                save_file_to_s3(merged_df, bucket_name, output_s3_key)

        else:
            metrics_filename = f"metrics_output.csv"
            logger.info(f"Saving results locally to: {metrics_path}")
            metrics_path = os.path.join(output_dir, metrics_filename)
            df.to_csv(metrics_path, index=False)


            if bucket_name:
                logger.info("Also uploading local results to S3.")
                save_file_to_s3(merged_df, bucket_name, f"{output_s3_key}")

    except Exception as e:
        logger.error(f"Error in saving results: {e}")
        raise


def lambda_handler(event, context):
    for k, v in event.items():
        os.environ[k] = str(v)

    is_lambda = os.environ.get('IS_LAMBDA', 'false').lower() == 'true'
    bucket_name = os.getenv("BUCKET_NAME_SENTIMENT","roberta-data-text")
    file_key =os.getenv("INPUT_KEY_SENTIMENT","channel_partner_feedback.csv")
    file_key_2 =os.getenv("INPUT_KEY_STOCKIST","augmented_stockist_data_with_sentiment_cleaned.csv")
    evaluation_output_key = os.getenv("SENTIMENT_EVALUATION_OUTPUT_KEY", "results/sentiment_evaluation_metrics.csv")
    output_s3_key=os.getenv("OUPUT_S3_KEY","results/gpt_model_output.csv")

        
    # Get API key from .env
    openai.api_key = os.getenv("OPENAI_API_KEY")
    logger.info("API Key:", openai.api_key)
    
    # Check if API key is loaded
    if not openai.api_key:
        raise ValueError("API key not found.")
        
    # Initialize S3 client
    s3 = boto3.client('s3')
    # Download file from S3
    local_file_path = 'new_channel_partner_feedback.csv'

    # Load data
    df = load_file_from_s3(bucket_name,file_key)
    stockist_df = load_file_from_s3(bucket_name,file_key_2)

    if "Feedback_Text" not in df.columns or "Sentiment" not in df.columns:
        raise ValueError("CSV file must contain 'Feedback_Text' and 'Sentiment' columns")

    # Sentiment prediction
    batch_size = 50
    gpt_predictions = []
    sentiment_scores = []

    for start in range(0, len(df), batch_size):
      batch_texts = df["Feedback_Text"].iloc[start:start+batch_size].tolist()
      batch_sentiments, batch_scores = get_gpt_batch_sentiment_with_score(batch_texts, batch_size=batch_size)
      gpt_predictions.extend(batch_sentiments)
      sentiment_scores.extend(batch_scores)
      # Sleep to avoid rate limiting between batches
      time.sleep(1)


    # Add predictions to dataframe
    df["gpt_prediction"] = gpt_predictions
    df["Feedback_Score"] = sentiment_scores

    # Evaluate
    logger.info("\nGPT Model Performance:")
    report = classification_report(df["Sentiment"], df["roberta_model_prediction"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    logger.info(report_df)

    save_results_to_csv_wrapper(report_df,evaluation_output_key,bucket_name=bucket_name, is_lambda)
    merged_df = pd.merge(stockist_df, df, on='Partner_id', how='left')
    # Save results
    # output_file_path = 'gpt_output.csv'
    # df.to_csv(output_file_path,index='false')

    # Upload to S3
    #output_s3_key = 'results/gpt_output.csv'
    
    save_results_to_csv_wrapper(merged_df,output_s3_key,bucket_name=bucket_name, is_lambda)
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
    event={"simulate": "run"}
    result = lambda_handler(event, None)
    logger.info(f"Execution result: {result}")
