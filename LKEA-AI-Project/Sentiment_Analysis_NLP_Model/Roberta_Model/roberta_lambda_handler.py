import boto3
import pandas as pd
from io import BytesIO
import logging
import os
import sys
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import traceback


# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Check if running in Lambda or locally
is_lambda = os.environ.get('IS_LAMBDA', 'false').lower() == 'true'

base_dir = base_dir = os.getcwd() #os.path.dirname(__file__)  # Gets the current script's directory
# Configure log file path based on environment
log_file_path = '/tmp/app.log' if is_lambda else os.path.join(base_dir, "LKEA-AI-Project","Sentiment_Analysis_NLP_Model","Roberta_Model","output_folder","logs")
PREDICTIONS_PATH = os.path.join(base_dir, "LKEA-AI-Project","Sentiment_Analysis_NLP_Model","Roberta_Model", "output_folder")



# Add handlers if they don't exist
if not logger.handlers:
    # Create a formatter that includes line numbers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

    try:
        # Add file handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to set up file handler: {e}")

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info(f"Running in Lambda mode: {is_lambda}")

# Set directory paths based on environment
BASE_DIR = '/tmp' if is_lambda else '.'
# Ensure directories exist
os.makedirs(f'{BASE_DIR}/input_data', exist_ok=True)
os.makedirs(f'{BASE_DIR}/output_folder', exist_ok=True)

# Initialize S3 client
s3_client = boto3.client('s3')

# Define RoBERTa model
roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"

def analyze_roberta_sentiment(text, tokenizer, model, threshold=0.5):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        probs = softmax(logits, dim=-1).numpy()[0]
        negative_prob, neutral_prob, positive_prob = probs
        sentiment = "Positive" if positive_prob >= threshold else "Negative"
        score = positive_prob  # Use positive probability as the score
        return sentiment, score
    except Exception as e:
        logger.error(f"Error in RoBERTa sentiment analysis: {e}")
        return "Negative", 0.0  # Default values in case of error

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

def save_results_to_csv_wrapper(merged_df, output_s3_key, is_lambda, bucket_name=None):
    """
    Save results to local or S3 based on execution environment.
    """
    #output_dir = '/tmp' if is_lambda else r'C:\Users\291688\LKEA-Project\LK-DS\LKEA-AI-Project\Sentiment_Analysis_NLP_Model\Roberta_Model\output_folder'
    #os.makedirs(output_dir, exist_ok=True)

    try:
        if is_lambda:
            logger.info("Lambda environment detected — uploading to S3.")
            if bucket_name:
                save_file_to_s3(merged_df, bucket_name, output_s3_key)
        else:
            metrics_filename = f"Robertaa_Sentiment_Predictions.csv"
            metrics_path = os.path.join(output_dir, metrics_filename)
            logger.info(f"Saving results locally to: {metrics_path}")
            merged_df.to_csv(metrics_path, index=False)

    except Exception as e:
        logger.error(f"Error in saving results: {e}")
        raise

def save_outputs(df, classification_report_text, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)

    df_output_path = os.path.join(output_dir, "Roberta_sentiment_predictions.csv")
    report_output_path = os.path.join(output_dir, "Roberta_Classification_Report.txt")

    df.to_csv(df_output_path, index=False)
    logging.info(f"Saved DataFrame with predictions to {df_output_path}")

    # Handle the classification report based on type
    if isinstance(classification_report_text, str):
        report_content = classification_report_text
    else:
        # Convert DataFrame or dict to string if needed
        report_content = str(classification_report_text)
        
    with open(report_output_path, "w") as f:
        f.write(report_content)
    logging.info(f"Saved classification report to {report_output_path}")

def lambda_handler(event, context):
    # Set environment variables from event
    for k, v in event.items():
        os.environ[k] = str(v)

    is_lambda = os.environ.get('IS_LAMBDA', 'false').lower() == 'true'
    bucket_name = os.getenv("BUCKET_NAME_SENTIMENT", "roberta-data-text")
    
    # Fix file paths to handle local and S3 paths correctly
    file_key = os.getenv("INPUT_KEY_SENTIMENT")
    file_key_2 = os.getenv("INPUT_KEY_STOCKIST")
    
    # Default S3 output keys
    evaluation_output_key = os.getenv("SENTIMENT_EVALUATION_OUTPUT_KEY", "results/sentiment_evaluation_metrics.csv")
    output_s3_key = os.getenv("OUPUT_S3_KEY", "results/roberta_model_output.csv")
    
    try:
        # Initialize the RoBERTa model and tokenizer
        logger.info("Loading RoBERTa model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)
        model.eval()
        logger.info("RoBERTa model and tokenizer loaded successfully")
        
        # Load data based on environment
        if is_lambda:
            local_file_path = 'channel_partner_feedback.csv'
            local_file_path_2 = 'augmented_stockist_data.csv'
            
            # Download from S3 to Lambda tmp directory
            s3_client.download_file(bucket_name, file_key, local_file_path)
            s3_client.download_file(bucket_name, file_key_2, local_file_path_2)
            
            df = load_file_locally(local_file_path)
            stockist_df = load_file_locally(local_file_path_2)
        else:
            # Try loading from S3 first, fallback to local file if that fails
            try:
                df = load_file_from_s3(bucket_name, file_key)
                stockist_df = load_file_from_s3(bucket_name, file_key_2)
            except Exception as e:
                logger.warning(f"Failed to load from S3, trying local file: {e}")
                # Try handling file_key as a local file path
                df = load_file_locally(file_key)
                stockist_df = load_file_locally(file_key_2)

        # Validate required columns
        if "Feedback_Text" not in df.columns or "Sentiment" not in df.columns:
            raise ValueError("CSV file must contain 'Feedback_Text' and 'Sentiment' columns")

        # Sentiment prediction
        logger.info("Running Roberta sentiment analysis...")
        roberta_predictions = []
        sentiment_scores = []
        
        # Process each feedback text
        for text in df["Feedback_Text"]:
            sentiment, score = analyze_roberta_sentiment(text, tokenizer, model)
            roberta_predictions.append(sentiment)
            sentiment_scores.append(score)
            
        # Add predictions to dataframe
        df["roberta_prediction"] = roberta_predictions
        df["Feedback_Score"] = sentiment_scores

        # Evaluate
        logger.info("\nRoBERTa Model Performance:")
        report = classification_report(df["Sentiment"], df["roberta_prediction"], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        logger.info(report_df)

        # Save evaluation metrics
        save_results_to_csv_wrapper(report_df, evaluation_output_key, is_lambda, bucket_name=bucket_name)
        
        # Merge dataframes
        merged_df = pd.merge(stockist_df, df, on='Partner_id', how='left')

        # Save outputs
        #os.makedirs(f"{BASE_DIR}/Roberta_Model/output_folder", exist_ok=True)
        save_outputs(merged_df, report_df, PREDICTIONS_PATH)
        
        classification_report_str = report_df.to_dict()

        return {
            'statusCode': 200,
            'body': {
                'message': 'Sentiment analysis completed successfully',
                'classification_report': classification_report_str,
                's3_output_path': f"s3://{bucket_name}/{output_s3_key}" if bucket_name else "Local file saved"
            }
        }
        
    except Exception as e:
        error_message = f"Error in lambda_handler: {str(e)}"
        logger.error(error_message)
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': {'error': error_message}
        }


if __name__ == "__main__":
    os.environ['IS_LAMBDA'] = 'false'
    base_dir = os.getcwd()  # Gets the current directory    
    # Define the new paths
    feedback_csv = os.path.join(base_dir, "LKEA-AI-Project","Sentiment_Analysis_NLP_Model", "input_data", "new_channel_partner_feedback.csv")
    stockist_csv = os.path.join(base_dir, "LKEA-AI-Project","Sentiment_Analysis_NLP_Model", "input_data", "Augmented_Stockist_Data_Final.csv")
    
    # Set environment variables
    os.environ['INPUT_KEY_SENTIMENT'] = feedback_csv
    os.environ['INPUT_KEY_STOCKIST'] = stockist_csv
    
    logger.info("Starting execution of lambda_handler")
    event = {"simulate": "run"}
    result = lambda_handler(event, None)
    logger.info(f"Execution result: {result}")