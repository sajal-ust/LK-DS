

import stanza
import pandas as pd
from sklearn.metrics import classification_report

# Load Stanza NLP Model
stanza.download("en")
nlp = stanza.Pipeline(lang="en", processors="tokenize,sentiment")

def get_stanza_sentiment(text, nlp):
    doc = nlp(text)
    sentiment_score = doc.sentences[0].sentiment  # (0 = Negative,1 = Positive)
    return "Positive" if sentiment_score == 2 else "Negative"

# Load Data
df = pd.read_csv("channel_partner_feedback.csv")  # Ensure it has 'text' and 'true_label' columns

# Analyze Sentiment for Stanza
df["stanza_prediction"] = df["Feedback_Text"].apply(lambda x: get_stanza_sentiment(x, nlp))

# Print classification report
print("\n Stanza Model Performance:")
print(classification_report(df["Sentiment"], df["stanza_prediction"]))
df.to_csv("stanza_output.csv", index=False)
df















