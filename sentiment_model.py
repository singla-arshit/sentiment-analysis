from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

labels = ["Negative", "Neutral", "Positive"]

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
    return {
        "text": text,
        "sentiment": labels[predicted_class.item()],
        "confidence": round(confidence.item(), 4),
        "probabilities": {
            labels[i]: round(probs[0][i].item(), 4)
            for i in range(len(labels))
        }
    }
