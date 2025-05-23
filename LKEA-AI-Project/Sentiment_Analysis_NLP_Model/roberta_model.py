

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import pandas as pd
from sklearn.metrics import classification_report

# Load RoBERTa Sentiment Analysis Model
roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)
roberta_model.eval()

def analyze_roberta_sentiment(text, tokenizer, model, threshold=0.5):
    """
    Analyze sentiment using RoBERTa and classify as Positive, or Negative.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = roberta_model(**inputs)
        logits = outputs.logits

    probs = softmax(logits, dim=-1).numpy()[0]  # Convert logits to probabilities
    negative_prob, neutral_prob, positive_prob = probs  # Extract individual probabilities

    # Classify based on confidence threshold
    if positive_prob >= threshold:
        return "Positive"
    else:
        return "Negative"

# Load Data
df = pd.read_csv("channel_partner_feedback.csv")  # Ensure it has 'text' and 'true_label' columns

# Analyze Sentiment for RoBERTa
df["roberta_model_prediction"] = df["Feedback_Text"].apply(lambda x: analyze_roberta_sentiment(x, tokenizer, roberta_model))

# Print classification report
print("\n RoBERTa Model Performance:")
print(classification_report(df["Sentiment"], df["roberta_model_prediction"]))

df



