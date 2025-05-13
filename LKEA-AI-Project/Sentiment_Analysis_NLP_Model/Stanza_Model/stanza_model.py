
import stanza
import pandas as pd
import logging
import os
from sklearn.metrics import classification_report

def setup_logger(log_file_path):
    """Set up a logger that writes to a file and creates the log directory if missing."""
    logger = logging.getLogger("stanza_sentiment")
    logger.setLevel(logging.INFO)

    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists

    if not logger.handlers:
        fh = logging.FileHandler(log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def download_stanza_model():
    """Ensure Stanza English model is downloaded."""
    stanza.download("en", verbose=False)

def load_stanza_pipeline():
    """Load the Stanza NLP pipeline."""
    return stanza.Pipeline(lang="en", processors="tokenize,sentiment", verbose=False)

def get_stanza_sentiment(text, nlp):
    """Classify sentiment as Positive or Negative using Stanza."""
    try:
        doc = nlp(text)
        sentiment_score = doc.sentences[0].sentiment  # 0 = Negative, 1 = Neutral, 2 = Positive
        return "Positive" if sentiment_score == 2 else "Negative"
    except Exception as e:
        return "Negative"  # fallback in case of error

def run_stanza_pipeline(feedback_csv, stockist_csv, prediction_output_path, evaluation_txt_path, logger):
    logger.info("Loading input data...")
    df = pd.read_csv(feedback_csv)
    stockist_df = pd.read_csv(stockist_csv)

    logger.info("Running Stanza sentiment analysis...")
    download_stanza_model()
    nlp = load_stanza_pipeline()

    df["stanza_prediction"] = df["Feedback_Text"].apply(lambda x: get_stanza_sentiment(x, nlp))

    logger.info("Generating classification report...")
    report = classification_report(df["Sentiment"], df["stanza_prediction"])
    logger.info("\n" + report)

    logger.info("Saving evaluation report to text file...")
    os.makedirs(os.path.dirname(evaluation_txt_path), exist_ok=True)
    with open(evaluation_txt_path, "w") as f:
        f.write(report)

    logger.info("Merging predictions with stockist data...")
    merged_df = pd.merge(stockist_df, df, on="Partner_id", how="left")

    logger.info("Saving merged output to CSV...")
    os.makedirs(os.path.dirname(prediction_output_path), exist_ok=True)
    merged_df.to_csv(prediction_output_path, index=False)
    logger.info(f"Prediction results saved to {prediction_output_path}")

    logger.info("Pipeline completed successfully.")
    return merged_df

# Assume you're in the Sentiment_Analysis_NLP_Model directory
base_dir = base_dir = os.getcwd() #os.path.dirname(__file__)  # Gets the current script's directory
feedback_csv = os.path.join(base_dir,"LKEA-AI-Project","Sentiment_Analysis_NLP_Model", "input_data", "new_channel_partner_feedback.csv")
stockist_csv = os.path.join(base_dir,"LKEA-AI-Project","Sentiment_Analysis_NLP_Model", "input_data", "Augmented_Stockist_Data_Final.csv")
prediction_output = os.path.join(base_dir, "LKEA-AI-Project","Sentiment_Analysis_NLP_Model","Stanza_Model", "outputs", "stanza_predictions.csv")
evaluation_output = os.path.join(base_dir, "LKEA-AI-Project","Sentiment_Analysis_NLP_Model","Stanza_Model", "outputs", "stanza_evaluation_report.txt")
log_file = os.path.join(base_dir, "LKEA-AI-Project","Sentiment_Analysis_NLP_Model","Stanza_Model", "outputs", "stanza_pipeline.log")
# Setup logger
logger = setup_logger(log_file)

try:
    run_stanza_pipeline(feedback_csv, stockist_csv, prediction_output, evaluation_output, logger)
except Exception as e:
    logger.exception("Pipeline failed due to an error.")












