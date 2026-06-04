# ==============================================================================
# Project: FastAPI Routing Interface for Sentiment Analysis Engine
# Author: Zaid Seliya
# UIN: 231A050 
# Department of Computer Engineering
# Rizvi College of Engineering
# ==============================================================================

from fastapi import FastAPI
from pydantic import BaseModel
from src.model import BERT_Sentiment_Classifier

app = FastAPI(
    title="Sentiment Analysis Engine API",
    description="Real-time sentiment classifier for social media monitoring using fine-tuned BERT transformers.",
    version="1.0.0"
)

model_engine = BERT_Sentiment_Classifier()

class TextPayload(BaseModel):
    text: str

@app.get("/")
def root_endpoint():
    return {
        "status": "Online",
        "project": "Sentiment Analysis Engine (NLP)",
        "developer": "Zaid Seliya",
        "college": "Rizvi College of Engineering",
        "docs_url": "/docs"
    }

@app.post("/predict")
def predict_sentiment(payload: TextPayload):
    if not payload.text.strip():
        return {"error": "Input text payload cannot be empty."}
        
    result = model_engine.analyze_text(payload.text)
    return result
  
