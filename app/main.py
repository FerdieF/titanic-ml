# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

# Path ke model dari train.py
MODEL_PATH = os.path.join("models", "model.pkl")

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict survival of Titanic passengers using a trained RandomForest model.",
    version="1.0.0",
)


# Schema input dari user (request body)
class PassengerInput(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str = Field(..., pattern="^(male|female)$")
    Age: float = Field(..., ge=0)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Embarked: str = Field(..., pattern="^(C|Q|S)$")

class PredictionOutput(BaseModel):
    input: dict
    survived: int
    probability_survived: float


# Load model saat aplikasi start
@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run train.py first.")
    model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {"message": "Titanic Survival Prediction API is running"}


@app.post("/predict", response_model=PredictionOutput)
def predict_survival(passenger: PassengerInput):
    # Convert input ke DataFrame untuk diproses oleh Pipeline sklearn
    input_df = pd.DataFrame([passenger.dict()])

    # Predict probability dan label
    proba = model.predict_proba(input_df)[0][1]   # prob. Survived = 1
    pred = int(proba >= 0.5)

    return {
        "input": passenger.dict(),
        "survived": pred,
        "probability_survived": proba,
    }
