"""
Preprocessing functions for churn prediction model.
Handles data cleaning, feature engineering, and transformation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, BINARY_MAPPING


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric and handle missing values."""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill NaN with 0 (customers with 0 tenure)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features from raw data."""
    df = df.copy()
    
    # Service count features
    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    df['num_services'] = 0
    for service in services:
        if service in df.columns:
            df['num_services'] += (df[service] == 'Yes').astype(int)
    
    # Tenure-based features
    df['is_new_customer'] = (df['tenure'] <= 12).astype(int)
    df['is_long_term'] = (df['tenure'] > 48).astype(int)
    
    # Contract features
    df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
    
    # Payment features
    df['uses_electronic_check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['uses_auto_payment'] = df['PaymentMethod'].apply(
        lambda x: 1 if 'automatic' in str(x).lower() else 0
    )
    
    # Internet features
    df['has_fiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['has_internet'] = (df['InternetService'] != 'No').astype(int)
    
    # Charges per service
    df['charges_per_service'] = df['MonthlyCharges'] / (df['num_services'] + 1)
    
    # Average monthly spend
    df['avg_monthly_spend'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )
    
    return df


def encode_categorical_features(df: pd.DataFrame, 
                                 label_encoders: Dict[str, LabelEncoder] = None,
                                 fit: bool = True) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode categorical features using LabelEncoder."""
    df = df.copy()
    
    if label_encoders is None:
        label_encoders = {}
    
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
            
        if fit:
            le = LabelEncoder()
            # Fit with all possible values
            df[col] = df[col].astype(str)
            le.fit(df[col])
            label_encoders[col] = le
            df[col] = le.transform(df[col])
        else:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].astype(str)
                # Handle unseen categories
                df[col] = df[col].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])
    
    return df, label_encoders


def preprocess_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Full preprocessing pipeline for training data.
    
    Returns:
        X: Feature matrix
        y: Target variable
        artifacts: Dict containing encoders and scaler for inference
    """
    # Clean data
    df = clean_total_charges(df)
    
    # Create target variable
    y = (df['Churn'] == 'Yes').astype(int)
    
    # Create features
    df = create_features(df)
    
    # Encode categorical features
    df, label_encoders = encode_categorical_features(df, fit=True)
    
    # Select final features
    feature_cols = (
        NUMERIC_FEATURES + 
        CATEGORICAL_FEATURES + 
        ['num_services', 'is_new_customer', 'is_long_term', 
         'is_month_to_month', 'uses_electronic_check', 'uses_auto_payment',
         'has_fiber', 'has_internet', 'charges_per_service', 'avg_monthly_spend']
    )
    
    # Only keep columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                    'charges_per_service', 'avg_monthly_spend', 'num_services']
    numeric_cols = [col for col in numeric_cols if col in X.columns]
    
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    artifacts = {
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'numeric_cols': numeric_cols
    }
    
    return X, y, artifacts


def preprocess_single_input(data: Dict[str, Any], artifacts: Dict) -> pd.DataFrame:
    """
    Preprocess a single input for prediction.
    
    Args:
        data: Dictionary with customer features
        artifacts: Preprocessing artifacts from training
    
    Returns:
        Feature vector ready for prediction
    """
    # Create DataFrame from input
    df = pd.DataFrame([data])
    
    # Handle missing fields with defaults
    defaults = {
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'No',
        'OnlineSecurity': 'No internet service',
        'OnlineBackup': 'No internet service',
        'DeviceProtection': 'No internet service',
        'TechSupport': 'No internet service',
        'StreamingTV': 'No internet service',
        'StreamingMovies': 'No internet service',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 0.0,
        'TotalCharges': 0.0
    }
    
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val
    
    # Clean and create features
    df = clean_total_charges(df)
    df = create_features(df)
    
    # Encode categorical features using saved encoders
    df, _ = encode_categorical_features(
        df, 
        label_encoders=artifacts['label_encoders'],
        fit=False
    )
    
    # Select features in correct order
    feature_cols = artifacts['feature_cols']
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols]
    
    # Scale numeric features
    numeric_cols = artifacts['numeric_cols']
    X[numeric_cols] = artifacts['scaler'].transform(X[numeric_cols])
    
    return X
