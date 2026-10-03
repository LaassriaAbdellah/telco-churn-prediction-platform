"""Transformation unique des entrées client pour Streamlit et FastAPI."""
import pandas as pd

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
VALID_VALUES = {
    "contract": {"Month-to-month", "One year", "Two year"},
    "internet_service": {"DSL", "Fiber optic", "No"},
    "tech_support": {"Yes", "No", "No internet service"},
    "online_security": {"Yes", "No", "No internet service"},
    "paperless_billing": {"Yes", "No"},
    "payment_method": {"Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"},
}


def build_client_frame(data, model_columns, scaler):
    for field, allowed in VALID_VALUES.items():
        if data[field] not in allowed:
            raise ValueError(f"Valeur invalide pour {field}: {data[field]}")
    row = {
        "tenure": data["tenure"],
        "MonthlyCharges": data["monthly_charges"],
        "TotalCharges": data["total_charges"],
        f"Contract_{data['contract']}": 1,
        f"InternetService_{data['internet_service']}": 1,
        f"TechSupport_{data['tech_support']}": 1,
        f"OnlineSecurity_{data['online_security']}": 1,
        "PaperlessBilling_Yes": int(data["paperless_billing"] == "Yes"),
        f"PaymentMethod_{data['payment_method']}": 1,
    }
    frame = pd.DataFrame([row]).reindex(columns=model_columns, fill_value=0)
    frame[NUMERIC_COLUMNS] = scaler.transform(frame[NUMERIC_COLUMNS])
    return frame
