import os
import joblib
import pandas as pd
from typing import Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="REST API for predicting customer churn using a trained Machine Learning Pipeline",
    version="1.0.0"
)

# Global model container
model_pipeline = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "churn_model.pkl")

@app.on_event("startup")
def load_model():
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            print(f"Model pipeline successfully loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {str(e)}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. API will fail predictions until trained model is placed.")


class CustomerData(BaseModel):
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=1)
    PhoneService: str = Field(..., example="No")
    MultipleLines: str = Field(..., example="No phone service")
    InternetService: str = Field(..., example="DSL")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=29.85)
    TotalCharges: Union[float, str] = Field(..., example=29.85)


class PredictionOutput(BaseModel):
    prediction: str = Field(..., example="Yes")
    churn_probability: float = Field(..., example=0.82)


def preprocess_input(input_data: CustomerData) -> pd.DataFrame:
    """Converts Pydantic CustomerData model to DataFrame and applies feature engineering."""
    data_dict = input_data.model_dump()
    df = pd.DataFrame([data_dict])
    
    # Handle TotalCharges string to float conversion if necessary
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    
    # Feature Engineering
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["TotalServices"] = df[service_cols].apply(lambda row: sum(1 for val in row if str(val).strip() == "Yes"), axis=1)
    
    # Tenure Grouping
    def get_tenure_group(t):
        if t <= 12:
            return "0-12m"
        elif t <= 24:
            return "13-24m"
        elif t <= 48:
            return "25-48m"
        else:
            return "49-72m"
            
    df["TenureGroup"] = df["tenure"].apply(get_tenure_group)
    
    # Automatic Payment Flag
    df["AutomaticPayment"] = df["PaymentMethod"].apply(lambda x: 1 if "automatic" in str(x).lower() else 0)
    
    return df


@app.get("/")
def root():
    return {
        "message": "Telco Customer Churn Prediction API",
        "status": "active",
        "docs": "/docs",
        "model_loaded": model_pipeline is not None
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerData):
    global model_pipeline
    
    if model_pipeline is None:
        # Reload model if missing
        if os.path.exists(MODEL_PATH):
            model_pipeline = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=500, detail="Trained model pipeline not found. Please train and save the model first.")
            
    try:
        df_processed = preprocess_input(customer)
        
        # Get churn prediction and probability
        pred_class = model_pipeline.predict(df_processed)[0]
        pred_proba = model_pipeline.predict_proba(df_processed)[0][1]
        
        prediction_label = "Yes" if pred_class == 1 or str(pred_class).lower() == "yes" else "No"
        
        return PredictionOutput(
            prediction=prediction_label,
            churn_probability=round(float(pred_proba), 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")
