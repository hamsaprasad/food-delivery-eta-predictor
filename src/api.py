import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.database import init_db, insert_prediction, get_recent_predictions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "eta_model.joblib")

# Global model holder
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure database and model are ready
    init_db()
    get_model()
    yield
    # Clean shutdown if needed

app = FastAPI(
    title="Food Delivery ETA Prediction API",
    description="Predict delivery ETA and detect delay risk using XGBoost",
    version="1.0.0",
    lifespan=lifespan
)

class DeliveryFeatures(BaseModel):
    # 10 Numeric Features
    Order_Hour: int = Field(default=12, description="Hour of the day (0-23)", ge=0, le=23)
    Is_Weekend: int = Field(default=0, description="1 if weekend, 0 otherwise", ge=0, le=1)
    Is_Festival: int = Field(default=0, description="1 if festival day, 0 otherwise", ge=0, le=1)
    Rider_Experience_Years: float = Field(default=3.5, description="Years of rider experience", ge=0.0)
    Rider_Rating: float = Field(default=4.2, description="Rider rating (1.0 - 5.0)", ge=1.0, le=5.0)
    Restaurant_Rating: float = Field(default=4.0, description="Restaurant rating (1.0 - 5.0)", ge=1.0, le=5.0)
    Order_Items: int = Field(default=2, description="Number of items in the order", ge=1)
    Preparation_Time_Min: int = Field(default=15, description="Preparation time in minutes", ge=1)
    Road_Distance_km: float = Field(default=12.5, description="Delivery distance in km", ge=0.0)
    Number_of_Signals: int = Field(default=6, description="Traffic signals on the route", ge=0)

    # 10 Categorical Features
    Day_of_Week: str = Field(default="Monday", description="Day of the week")
    Weather: str = Field(default="Clear", description="Weather condition (Clear, Rainy, Stormy, etc.)")
    Pickup_Zone: str = Field(default="CBD", description="Pickup Zone")
    Dropoff_Zone: str = Field(default="Residential", description="Dropoff Zone")
    Vehicle_Type: str = Field(default="Bike", description="Vehicle type (Bike, Scooter, Car, etc.)")
    Cuisine_Type: str = Field(default="North Indian", description="Cuisine type")
    Restaurant_Load: str = Field(default="Medium", description="Restaurant load (Low, Medium, High)")
    Delivery_Distance_Category: str = Field(default="Medium", description="Distance category (Short, Medium, Long)")
    Traffic_Level: str = Field(default="Moderate", description="Traffic level (Low, Moderate, High, Jam)")
    Delivery_Priority: str = Field(default="Normal", description="Priority (Normal, High, Urgent)")

class PredictionResponse(BaseModel):
    estimated_minutes: float
    delay_risk: str
    record_id: int
    message: str

@app.get("/")
def root():
    return {"status": "ok", "message": "Food Delivery ETA Prediction API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_eta(features: DeliveryFeatures):
    eta_pipeline = get_model()

    # Convert Pydantic model to DataFrame for pipeline prediction
    input_data = pd.DataFrame([features.model_dump()])

    try:
        raw_pred = float(eta_pipeline.predict(input_data)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    estimated_minutes = round(max(0.0, raw_pred), 2)
    delay_risk = "High" if estimated_minutes > 90 else "Normal"

    # Persist prediction to SQLite
    record_id = insert_prediction(
        road_distance_km=features.Road_Distance_km,
        traffic_level=features.Traffic_Level,
        weather=features.Weather,
        vehicle_type=features.Vehicle_Type,
        preparation_time_min=features.Preparation_Time_Min,
        predicted_eta_mins=estimated_minutes,
        delay_risk=delay_risk
    )

    return PredictionResponse(
        estimated_minutes=estimated_minutes,
        delay_risk=delay_risk,
        record_id=record_id,
        message="Prediction successfully generated and saved to database"
    )

@app.get("/history")
def get_prediction_history(limit: int = 10):
    """Retrieve recent predictions from the SQLite database."""
    history = get_recent_predictions(limit=limit)
    return {"count": len(history), "predictions": history}
