
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Used Car Price Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model.pkl")


class CarDetails(BaseModel):
    brand: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Used Car Price Predictor API is running"
    }


@app.post("/predict")
def predict_price(car: CarDetails):
    input_data = pd.DataFrame([{
        "brand": car.brand,
        "year": car.year,
        "km_driven": car.km_driven,
        "fuel": car.fuel,
        "seller_type": car.seller_type,
        "transmission": car.transmission,
        "owner": car.owner
    }])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_price": round(float(prediction), 2),
        "currency": "INR"
    }
