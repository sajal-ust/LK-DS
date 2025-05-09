import pandas as pd
import time
import os
import openai
import logging
from dotenv import load_dotenv
from sklearn.metrics import classification_report

# Custom output paths
OUTPUT_DIR = r"C:\Users\291688\LKEA-Project\LK-DS\LKEA-AI-Project\Sentiment_Analysis_NLP_Model\GPT_3.5_Model\outputs"
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "gpt_predictions.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "GPT_classification_report.txt")
LOG_PATH = os.path.join(OUTPUT_DIR, "logs.log")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
def load_api_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Please check your .env file.")
        raise ValueError("OpenAI API key not found.")
    openai.api_key = api_key
    logger.info("OpenAI API key loaded successfully.")

# Sentiment batch processor
def get_gpt_batch_sentiment_with_score(texts, batch_size=50, timeout=10):
    # Make sure we have texts to process
    if not texts:
        logger.warning("Empty text batch received")
        return [], []
    
    batched_prompt = "\n\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
    message_content = (
        "Classify each of the following texts as Positive or Negative and provide a sentiment score between 0 and 1 "
        "(higher value means more positive). Return the results in the format '1: Positive, 0.85'.\n\n"
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
        
        # Check if response contains expected data
        if not response.get('choices') or len(response['choices']) == 0:
            logger.error(f"Unexpected API response format: {response}")
            return ["Negative"] * len(texts), [0.0] * len(texts)
            
        output = response['choices'][0]['message']['content'].strip()
        
        # Debug the response
        logger.debug(f"API response: {output}")
        
        sentiments = []
        scores = []
        
        # Parse each line of the response
        for i, line in enumerate(output.splitlines()):
            if i >= len(texts):  # Ensure we don't exceed the number of input texts
                break
                
            try:
                # Extract sentiment and score from the line
                try:
                    # First remove the line number prefix (handles both "1:" and "1." formats)
                    # Find the position where the actual content starts
                    number_end_pos = -1
                    for char_pos, char in enumerate(line):
                        if char in [":", "."]:
                            number_end_pos = char_pos
                            break
                    
                    if number_end_pos >= 0:
                        # Extract the content after the number prefix
                        content = line[number_end_pos + 1:].strip()
                        
                        # Split by comma to get sentiment and score
                        if "," in content:
                            sentiment_text, score_text = content.split(",", 1)
                            sentiment = sentiment_text.strip()
                            
                            # Normalize sentiment to "Positive" or "Negative"
                            if "positive" in sentiment.lower():
                                sentiment = "Positive"
                            else:
                                sentiment = "Negative"
                            
                            # Extract score
                            try:
                                score = float(score_text.strip())
                                # Ensure score is between 0 and 1
                                score = max(0.0, min(1.0, score))
                            except ValueError:
                                logger.debug(f"Could not convert score to float: {score_text}")
                                score = 0.75 if sentiment == "Positive" else 0.25
                        else:
                            # No score provided, just sentiment
                            sentiment = content
                            if "positive" in sentiment.lower():
                                sentiment = "Positive"
                                score = 0.75
                            else:
                                sentiment = "Negative"
                                score = 0.25
                    else:
                        # No number prefix found, try to extract sentiment directly
                        if "positive" in line.lower():
                            sentiment = "Positive"
                            score = 0.75
                        else:
                            sentiment = "Negative"
                            score = 0.25
                    
                    sentiments.append(sentiment)
                    scores.append(score)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse line: {line}, Error: {e}")
                    if "positive" in line.lower():
                        sentiments.append("Positive")
                        scores.append(0.75)
                    else:
                        sentiments.append("Negative")
                        scores.append(0.25)
            except Exception as e:
                logger.warning(f"Failed to parse line: {line}, Error: {e}")
                sentiments.append("Negative")
                scores.append(0.0)
        
        # Handle case where OpenAI returns fewer responses than expected
        if len(sentiments) < len(texts):
            logger.warning(f"Received {len(sentiments)} responses for {len(texts)} texts")
            # Fill in missing responses
            sentiments.extend(["Negative"] * (len(texts) - len(sentiments)))
            scores.extend([0.0] * (len(texts) - len(scores)))
        
        return sentiments, scores
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return ["Negative"] * len(texts), [0.0] * len(texts)

# Main processing function
def process_feedback_data(feedback_path, stockist_path, batch_size=20):
    try:
        df = pd.read_csv(feedback_path)
        stockist_df = pd.read_csv(stockist_path)
        
        # Check if dataframes are loaded correctly
        if df.empty or stockist_df.empty:
            logger.error("One or both input files are empty")
            return None
            
        # Ensure Feedback_Text column exists and has data
        if "Feedback_Text" not in df.columns:
            logger.error("Feedback_Text column not found in input data")
            return None
            
        # Check for null values in Feedback_Text
        null_count = df["Feedback_Text"].isna().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} null values in Feedback_Text column")
            # Replace null values with empty string
            df["Feedback_Text"] = df["Feedback_Text"].fillna("")

        gpt_predictions = []
        sentiment_scores = []

        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch_texts = df["Feedback_Text"].iloc[start:end].tolist()
            logger.info(f"Processing batch {start} to {end} ({len(batch_texts)} texts)")
            
            # Skip empty batches
            if not batch_texts:
                logger.warning(f"Empty batch from {start} to {end}")
                continue
                
            # Process batch with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    batch_sentiments, batch_scores = get_gpt_batch_sentiment_with_score(batch_texts, batch_size=batch_size)
                    break
                except Exception as e:
                    logger.error(f"Batch processing failed (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        # Use default values for this batch if all retries fail
                        batch_sentiments = ["Negative"] * len(batch_texts)
                        batch_scores = [0.0] * len(batch_texts)
                    time.sleep(2)  # Wait before retry
            
            # Ensure we have the right number of predictions
            if len(batch_sentiments) != len(batch_texts):
                logger.warning(f"Mismatch between batch size ({len(batch_texts)}) and predictions ({len(batch_sentiments)})")
                # Adjust predictions to match batch size
                if len(batch_sentiments) < len(batch_texts):
                    batch_sentiments.extend(["Negative"] * (len(batch_texts) - len(batch_sentiments)))
                    batch_scores.extend([0.0] * (len(batch_texts) - len(batch_scores)))
                else:
                    batch_sentiments = batch_sentiments[:len(batch_texts)]
                    batch_scores = batch_scores[:len(batch_texts)]
            
            gpt_predictions.extend(batch_sentiments)
            sentiment_scores.extend(batch_scores)
            
            # Log progress
            logger.info(f"Processed batch {start} to {end}: {len(batch_sentiments)} predictions")
            
            # Rate limiting
            time.sleep(1)

        # Ensure we have the right number of predictions for the entire dataset
        if len(gpt_predictions) != len(df):
            logger.warning(f"Mismatch between dataset size ({len(df)}) and total predictions ({len(gpt_predictions)})")
            if len(gpt_predictions) < len(df):
                gpt_predictions.extend(["Negative"] * (len(df) - len(gpt_predictions)))
                sentiment_scores.extend([0.0] * (len(df) - len(sentiment_scores)))
            else:
                gpt_predictions = gpt_predictions[:len(df)]
                sentiment_scores = sentiment_scores[:len(df)]

        df["gpt_batch_prediction"] = gpt_predictions
        df["gpt_sentiment_score"] = sentiment_scores

        if "Sentiment" in df.columns:
            logger.info("Generating classification report...")
            report = classification_report(df["Sentiment"], df["gpt_batch_prediction"])
            with open(REPORT_PATH, "w") as f:
                f.write(report)
            logger.info(f"Classification report saved to: {REPORT_PATH}")
            print("\nGPT Batch Model Performance:")
            print(report)

        # Save predictions before merging
        df.to_csv(os.path.join(OUTPUT_DIR, "gpt_predictions_only.csv"), index=False)
        
        # Ensure Partner_id exists in both dataframes
        if "Partner_id" not in df.columns or "Partner_id" not in stockist_df.columns:
            logger.error("Partner_id column missing in one or both input files")
            return df
            
        # Merge dataframes
        try:
            merged_df = pd.merge(stockist_df, df, on="Partner_id", how="left")
            merged_df.to_csv(PREDICTIONS_PATH, index=False)
            logger.info(f"Merged predictions saved to: {PREDICTIONS_PATH}")
            return merged_df
        except Exception as e:
            logger.error(f"Error merging dataframes: {e}")
            return df
            
    except Exception as e:
        logger.error(f"Error in process_feedback_data: {e}")
        return None

if __name__ == "__main__":
    try:
        load_api_key()
        feedback_file = r"C:\Users\291688\LKEA-Project\LK-DS\LKEA-AI-Project\Sentiment_Analysis_NLP_Model\input_data\new_channel_partner_feedback.csv"
        stockist_file = r"C:\Users\291688\LKEA-Project\LK-DS\LKEA-AI-Project\Sentiment_Analysis_NLP_Model\input_data\Augmented_Stockist_Data_Final.csv"
        
        # Check if files exist
        if not os.path.exists(feedback_file):
            logger.error(f"Feedback file not found: {feedback_file}")
            exit(1)
        if not os.path.exists(stockist_file):
            logger.error(f"Stockist file not found: {stockist_file}")
            exit(1)
            
        merged_result_df = process_feedback_data(feedback_file, stockist_file)
        
        if merged_result_df is not None:
            logger.info("Processing complete.")
            print(merged_result_df.head())
        else:
            logger.error("Processing failed.")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")