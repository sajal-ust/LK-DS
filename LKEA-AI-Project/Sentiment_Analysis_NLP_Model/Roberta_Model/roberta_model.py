import torch
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import os
import logging  # Added for logging


# Setup Logger         

def setup_logger(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logging.basicConfig(
        filename=log_file_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()
    return logger


# Load RoBERTa Model   

def load_roberta_model(model_name="cardiffnlp/twitter-roberta-base-sentiment"):
    logger.info("Loading RoBERTa model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    logger.info("RoBERTa model loaded successfully.")
    return tokenizer, model


# Sentiment Label + Score Prediction 

def analyze_roberta_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = softmax(logits, dim=-1).numpy()[0]
    negative_prob, neutral_prob, positive_prob = probs
    sentiment_score = float(positive_prob)
    sentiment_label = "Positive" if sentiment_score >= 0.5 else "Negative"
    return sentiment_label, sentiment_score


# Save Classification Report to TXT Format   

def save_classification_report(y_true, y_pred, output_txt_path):
    logger.info("Saving classification report to txt...")
    report = classification_report(y_true, y_pred)
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w") as f:
        f.write("RoBERTa Sentiment Analysis Classification Report\n")
        f.write("="*60 + "\n")
        f.write(report)
    logger.info(f"Classification report saved to {output_txt_path}")


# Run Full RoBERTa Inference    

def run_roberta_pipeline(feedback_csv_path,
                         stockist_csv_path,
                         prediction_output_path,
                         evaluation_txt_path):
    logger.info("Starting sentiment analysis pipeline...")

    tokenizer, model = load_roberta_model()

    try:
        df = pd.read_csv(feedback_csv_path)
        stockist_df = pd.read_csv(stockist_csv_path)
        logger.info("CSV files loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading CSV files: {e}")
        raise

    required_cols = ["Feedback_Text", "Sentiment", "Partner_id"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"Missing required columns in feedback CSV: {missing}")
        raise ValueError(f"Missing columns: {missing}")

    results = df["Feedback_Text"].apply(lambda x: analyze_roberta_sentiment(x, tokenizer, model))
    df["roberta_model_prediction"], df["sentiment_score"] = zip(*results)

    save_classification_report(df["Sentiment"], df["roberta_model_prediction"], evaluation_txt_path)

    merged_df = pd.merge(stockist_df, df, on='Partner_id', how='left')

    os.makedirs(os.path.dirname(prediction_output_path), exist_ok=True)
    merged_df.to_csv(prediction_output_path, index=False)
    logger.info(f"Prediction results saved to {prediction_output_path}")
    
    logger.info("Pipeline completed successfully.")
    return merged_df


# Run as a script 


# Assume you're in the Sentiment_Analysis_NLP_Model directory
base_dir = base_dir = os.getcwd() #os.path.dirname(__file__)  # Gets the current script's directory

feedback_csv = os.path.join(base_dir,"LKEA-AI-Project","Sentiment_Analysis_NLP_Model", "input_data", "new_channel_partner_feedback.csv")
stockist_csv = os.path.join(base_dir,"LKEA-AI-Project","Sentiment_Analysis_NLP_Model", "input_data", "Augmented_Stockist_Data_Final.csv")
prediction_output = os.path.join(base_dir, "LKEA-AI-Project","Sentiment_Analysis_NLP_Model","Roberta_Model", "outputs", "roberta_predictions.csv")
evaluation_output = os.path.join(base_dir, "LKEA-AI-Project","Sentiment_Analysis_NLP_Model","Roberta_Model", "outputs", "roberta_evaluation_report.txt")
log_file = os.path.join(base_dir, "LKEA-AI-Project","Sentiment_Analysis_NLP_Model","Roberta_Model", "outputs", "roberta_pipeline.log")
# Setup logger
logger = setup_logger(log_file)

try:
    run_roberta_pipeline(feedback_csv, stockist_csv, prediction_output, evaluation_output)
except Exception as e:
    logger.exception("Pipeline failed due to an error.")
