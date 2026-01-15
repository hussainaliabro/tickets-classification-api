from fastapi import FastAPI
import pandas as pd
from app.schemas import EmailInput
from app.model_loader import model

app = FastAPI(
    title= "Customer Support Tickets Classification API",
    description= "Predict Queue, Language(English or German) and Priority of customer support tickets from subject and body of Email",
    version= "2026"
    )

@app.get("/")
def home():
    return {"message": "Customer Support Tickets Classifier API"}


@app.post('/predict')

def predict_tickets(data : EmailInput):
    
    input_df = pd.DataFrame([data.dict()])
    
    prediction = model.predict(input_df)
    
    return prediction