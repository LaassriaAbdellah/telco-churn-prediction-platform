"""API REST de prédiction du risque de churn."""
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from inference import build_client_frame

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
model = joblib.load(MODEL_DIR / "best_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
model_columns = joblib.load(MODEL_DIR / "model_columns.joblib")

app = FastAPI(title="API Prédiction Churn Telco")


class ClientData(BaseModel):
    tenure: int = Field(ge=0, le=72)
    monthly_charges: float = Field(gt=0, le=120)
    total_charges: float | None = Field(default=None, ge=0)
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    tech_support: Literal["Yes", "No", "No internet service"]
    online_security: Literal["Yes", "No", "No internet service"]
    paperless_billing: Literal["Yes", "No"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]


@app.post("/predict")
def predict_churn(data: ClientData):
    values = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    # Si le total réel manque, on conserve une estimation, explicitement signalée.
    estimated_total = values["total_charges"] is None
    if estimated_total:
        values["total_charges"] = values["tenure"] * values["monthly_charges"]
    frame = build_client_frame(values, model_columns, scaler)
    probability = float(model.predict_proba(frame)[0][1])
    percentage = round(probability * 100, 1)
    level = "High" if percentage >= 60 else "Medium" if percentage >= 30 else "Low"
    return {"churn_probability": percentage, "risk_level": level,
            "total_charges_estimated": estimated_total}
