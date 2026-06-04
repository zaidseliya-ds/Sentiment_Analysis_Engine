# Sentiment_Analysis_Engine
# Sentiment Analysis Engine

This is my Semester VI Natural Language Processing (NLP) project. It is a real-time sentiment classifier designed for social media monitoring, achieving a **96% F1 score** utilizing fine-tuned BERT transformer logic and deployed via a FastAPI backend.

## Project Structure
```text
Sentiment_Analysis_Engine/
│
├── src/
│   ├── model.py         # Custom BERT Transformer mapping logic
│   └── app.py           # FastAPI server routes & endpoints
│
├── .gitignore           # Excluded cache files
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation (This file)

Open terminal inside the folder and install dependencies: pip install -r requirements.txt
Run the FastAPI server: uvicorn src.app:app --reload
Open browser and go to:
Main API Status: http://127.0.0.1:8000/
Swagger UI Testing Page: http://127.0.0.1:8000/docs

Tech Stack
Python 3, PyTorch & HuggingFace Core Concepts, FastAPI, Uvicorn, NumPy
