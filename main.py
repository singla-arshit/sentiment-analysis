from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from sentiment_model import predict_sentiment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              #Kisi bhi origin se request allow ho (e.g. browser se)
    allow_credentials=True,
    allow_methods=["*"],              #POST, GET, etc. sab methods allow
    allow_headers=["*"],              #Sabhi headers allow
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def analyze_sentiment(data: TextInput):
    return predict_sentiment(data.text)
