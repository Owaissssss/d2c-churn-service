import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Initialize application
app = FastAPI(
    title="D2C Personal-Care Churn Scoring API",
    description="Live web service predicting customer churn risks using an optimized Random Forest Classifier model.",
    version="1.0.0"
)

# Load the trained model weights securely
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load serialized model binary: {str(e)}")

# Pre-load data signature columns from the snapshot file to guarantee an absolute fit match
try:
    base_df = pd.read_csv('rfm_modeling_snapshot.csv')
    drop_cols = ['customer_id', 'snapshot_date', 'churn_next_60d', 'split']
    cat_cols = ['city_tier', 'age_group', 'acquisition_channel', 'loyalty_tier', 'preferred_category',
                'marketing_consent']
    base_df_encoded = pd.get_dummies(base_df, columns=cat_cols, drop_first=True)
    MODEL_COLUMN_SIGNATURE = [c for c in base_df_encoded.columns if c not in drop_cols]
except Exception as e:
    raise RuntimeError(f"Failed to extract historical schema bounds: {str(e)}")


# Define the expected incoming JSON payload format using raw, human-readable strings
class CustomerFeatures(BaseModel):
    recency_days: float = Field(..., description="Days since the customer's last transaction.")
    frequency_180d: float = Field(..., description="Total orders placed in the last 180 days.")
    monetary_180d: float = Field(..., description="Gross revenue contributions over the last 180 days.")
    ticket_count_90d: float = Field(..., description="Total logged support desk tickets in the last 90 days.")
    sessions_30d: float = Field(..., description="Total web or mobile application session count in the last 30 days.")
    abandoned_carts_30d: float = Field(..., description="Total products added to shopping cart in the last 30 days.")
    city_tier: str = Field(..., description="E.g., Tier 1, Tier 2, Tier 3")
    age_group: str = Field(..., description="E.g., 18-24, 25-34, 35+")
    acquisition_channel: str = Field(...,
                                     description="E.g., Instagram, Organic, Influencer, Marketplace, Direct, Youtube")
    loyalty_tier: str = Field(..., description="E.g., Gold, Silver, Bronze, UNKNOWN_UNENROLLED")
    preferred_category: str = Field(..., description="E.g., Face, Hair, Body, Makeup")
    marketing_consent: str = Field(..., description="E.g., Yes, No")

    class Config:
        json_schema_extra = {
            "example": {
                "recency_days": 45.0,
                "frequency_180d": 6.0,
                "monetary_180d": 3500.0,
                "ticket_count_90d": 3.0,
                "sessions_30d": 22.0,
                "abandoned_carts_30d": 2.0,
                "city_tier": "Tier 2",
                "age_group": "25-34",
                "acquisition_channel": "Instagram",
                "loyalty_tier": "Gold",
                "preferred_category": "Face",
                "marketing_consent": "Yes"
            }
        }


@app.get("/")
@app.get("/health")
def health_check():
    """Operational gateway route verifying that the API service is healthy and online."""
    return {
        "status": "ok",
        "service": "D2C Churn Scoring Engine",
        "calibrated_threshold": 0.40
    }


def process_predictions(input_data_list: List[dict]) -> List[dict]:
    """Helper pipeline to run the tabular formatting, encoding, and thresholding steps."""
    # Convert list of JSON objects directly into a structured DataFrame
    temp_df = pd.DataFrame(input_data_list)

    # Apply dummy encoding
    temp_df_encoded = pd.get_dummies(temp_df, columns=cat_cols)

    # Re-index to ensure alignment with the matrix model weight signatures
    final_features_df = temp_df_encoded.reindex(columns=MODEL_COLUMN_SIGNATURE, fill_value=0)

    # Calculate probabilities
    raw_probabilities = model.predict_proba(final_features_df)[:, 1]

    results = []
    calibrated_threshold = 0.40

    for prob in raw_probabilities:
        churn_risk_score = float(prob)
        final_prediction = 1 if churn_risk_score >= calibrated_threshold else 0

        # Build risk explanations based on typical risk behavior flags
        explanation = "Standard behavior patterns."
        if churn_risk_score >= calibrated_threshold:
            explanation = "Elevated risk factors detected due to systemic behavior metrics or customer care friction."

        results.append({
            "churn_probability": round(churn_risk_score, 4),
            "predicted_class": final_prediction,
            "risk_level": "high" if final_prediction == 1 else "low",
            "risk_explanation": explanation
        })
    return results


@app.post("/predict")
def predict_churn(payload: CustomerFeatures):
    """Live single-record prediction inference endpoint mapping parameters straight to pipeline structures."""
    try:
        raw_input = payload.model_dump()
        prediction_output = process_predictions([raw_input])[0]
        return prediction_output
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error executing model pipeline scoring: {str(e)}")


@app.post("/batch_predict")
def batch_predict_churn(payload: List[CustomerFeatures]):
    """Live batch array prediction endpoint processing bulk matrices records seamlessly."""
    try:
        raw_inputs = [item.model_dump() for item in payload]
        batch_outputs = process_predictions(raw_inputs)
        return {"predictions": batch_outputs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error executing batch scoring pipeline: {str(e)}")