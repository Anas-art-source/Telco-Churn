#!/usr/bin/env python3
"""
app.py - FastAPI application for churn prediction.

Usage:
    uvicorn app:app --reload

Endpoints:
    GET  /health  - Health check
    POST /predict - Predict churn probability
"""

import os
import sys
from typing import Optional
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import preprocess_single_input
from src.config import VALID_CATEGORIES

# Global model and artifacts
model = None
artifacts = None


# ------------ Input Normalization Helpers ------------

def normalize_yes_no(value, field_name: str) -> str:
    """Normalize Yes/No fields - handles 1/0, true/false, case variations."""
    if isinstance(value, (int, float)):
        return "Yes" if value == 1 else "No"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, str):
        v_lower = value.strip().lower()
        if v_lower in ('yes', 'y', '1', 'true'):
            return "Yes"
        if v_lower in ('no', 'n', '0', 'false'):
            return "No"
    raise ValueError(
        f"Invalid value '{value}' for {field_name}. "
        f"Expected: 'Yes', 'No', 1, 0, true, or false"
    )


def normalize_categorical(value: str, field_name: str, valid_values: list) -> str:
    """Normalize categorical field with case-insensitive matching."""
    if not isinstance(value, str):
        raise ValueError(
            f"Invalid type for {field_name}: expected string, got {type(value).__name__}. "
            f"Valid values: {valid_values}"
        )
    
    # Try exact match first
    if value in valid_values:
        return value
    
    # Case-insensitive match
    value_lower = value.strip().lower()
    for valid in valid_values:
        if valid.lower() == value_lower:
            return valid
    
    # No match found
    raise ValueError(
        f"Invalid value '{value}' for {field_name}. "
        f"Valid values: {valid_values}"
    )


def normalize_internet_dependent(value, field_name: str) -> str:
    """Normalize fields that depend on internet service."""
    valid = ['Yes', 'No', 'No internet service']
    
    if isinstance(value, (int, float, bool)):
        return normalize_yes_no(value, field_name)
    
    if isinstance(value, str):
        v_lower = value.strip().lower()
        if v_lower in ('yes', 'y', '1', 'true'):
            return "Yes"
        if v_lower in ('no', 'n', '0', 'false'):
            return "No"
        if 'no internet' in v_lower or 'no_internet' in v_lower:
            return "No internet service"
    
    raise ValueError(
        f"Invalid value '{value}' for {field_name}. "
        f"Valid values: {valid}"
    )


def normalize_phone_dependent(value, field_name: str) -> str:
    """Normalize fields that depend on phone service."""
    valid = ['Yes', 'No', 'No phone service']
    
    if isinstance(value, (int, float, bool)):
        return normalize_yes_no(value, field_name)
    
    if isinstance(value, str):
        v_lower = value.strip().lower()
        if v_lower in ('yes', 'y', '1', 'true'):
            return "Yes"
        if v_lower in ('no', 'n', '0', 'false'):
            return "No"
        if 'no phone' in v_lower or 'no_phone' in v_lower:
            return "No phone service"
    
    raise ValueError(
        f"Invalid value '{value}' for {field_name}. "
        f"Valid values: {valid}"
    )


# ------------ Pydantic Models ------------

class CustomerInput(BaseModel):
    """Input schema for customer churn prediction."""
    
    gender: str = Field(default="Male", description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="Is senior citizen (0 or 1)")
    Partner: str = Field(default="No", description="Has partner (Yes/No)")
    Dependents: str = Field(default="No", description="Has dependents (Yes/No)")
    tenure: int = Field(default=1, ge=0, le=100, description="Months with company")
    PhoneService: str = Field(default="Yes", description="Has phone service (Yes/No)")
    MultipleLines: str = Field(default="No", description="Has multiple lines")
    InternetService: str = Field(default="DSL", description="Internet service type")
    OnlineSecurity: str = Field(default="No", description="Has online security")
    OnlineBackup: str = Field(default="No", description="Has online backup")
    DeviceProtection: str = Field(default="No", description="Has device protection")
    TechSupport: str = Field(default="No", description="Has tech support")
    StreamingTV: str = Field(default="No", description="Has streaming TV")
    StreamingMovies: str = Field(default="No", description="Has streaming movies")
    Contract: str = Field(default="Month-to-month", description="Contract type")
    PaperlessBilling: str = Field(default="Yes", description="Has paperless billing (Yes/No)")
    PaymentMethod: str = Field(default="Electronic check", description="Payment method")
    MonthlyCharges: float = Field(default=50.0, ge=0, description="Monthly charges")
    TotalCharges: Optional[float] = Field(default=None, ge=0, description="Total charges")
    
    @field_validator('gender', mode='before')
    @classmethod
    def validate_gender(cls, v):
        return normalize_categorical(v, 'gender', VALID_CATEGORIES['gender'])
    
    @field_validator('Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', mode='before')
    @classmethod
    def validate_yes_no_fields(cls, v, info):
        return normalize_yes_no(v, info.field_name)
    
    @field_validator('MultipleLines', mode='before')
    @classmethod
    def validate_multiple_lines(cls, v):
        return normalize_phone_dependent(v, 'MultipleLines')
    
    @field_validator('OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                     'TechSupport', 'StreamingTV', 'StreamingMovies', mode='before')
    @classmethod
    def validate_internet_dependent(cls, v, info):
        return normalize_internet_dependent(v, info.field_name)
    
    @field_validator('Contract', mode='before')
    @classmethod
    def validate_contract(cls, v):
        return normalize_categorical(v, 'Contract', VALID_CATEGORIES['Contract'])
    
    @field_validator('InternetService', mode='before')
    @classmethod
    def validate_internet(cls, v):
        return normalize_categorical(v, 'InternetService', VALID_CATEGORIES['InternetService'])
    
    @field_validator('PaymentMethod', mode='before')
    @classmethod
    def validate_payment(cls, v):
        return normalize_categorical(v, 'PaymentMethod', VALID_CATEGORIES['PaymentMethod'])

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 45.3,
                "TotalCharges": 543.6
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for churn prediction."""
    churn_probability: float = Field(description="Probability of churn (0-1)")
    prediction: str = Field(description="Prediction: 'Yes' or 'No'")
    risk_level: str = Field(description="Risk level: 'Low', 'Medium', or 'High'")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool
    version: str = "1.0.0"


# ------------ Lifespan Handler ------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, artifacts
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(models_dir, 'model.pkl')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        model = joblib.load(model_path)
        artifacts = joblib.load(preprocessor_path)
        print("✓ Model and preprocessor loaded successfully")
    else:
        print("⚠️ Warning: Model files not found. Run 'python train.py' first.")
    
    yield
    
    # Cleanup (if needed)
    print("Shutting down...")


# ------------ FastAPI App ------------

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predict customer churn probability using XGBoost",
    version="1.0.0",
    lifespan=lifespan
)


# ------------ Endpoints ------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(customer: CustomerInput):
    """
    Predict churn probability for a customer.
    
    Returns:
        - churn_probability: Float between 0 and 1
        - prediction: 'Yes' or 'No'
        - risk_level: 'Low' (< 0.3), 'Medium' (0.3-0.7), or 'High' (> 0.7)
    """
    if model is None or artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run 'python train.py' first."
        )
    
    # Convert input to dict
    input_data = customer.model_dump()
    
    # Handle TotalCharges default
    if input_data['TotalCharges'] is None:
        input_data['TotalCharges'] = input_data['MonthlyCharges'] * input_data['tenure']
    
    try:
        # Preprocess input
        X = preprocess_single_input(input_data, artifacts)
        
        # Get prediction
        probability = float(model.predict_proba(X)[0][1])
        prediction = "Yes" if probability >= 0.5 else "No"
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            churn_probability=round(probability, 4),
            prediction=prediction,
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/", tags=["Info"])
async def root():
    """API information."""
    return {
        "name": "Telco Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - Predict churn probability"
        },
        "docs": "/docs"
    }
