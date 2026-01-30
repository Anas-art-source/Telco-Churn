#!/usr/bin/env python3
"""
train.py - Train a churn prediction model using XGBoost with SMOTE.

Usage:
    python train.py

Output:
    - models/model.pkl: Trained XGBoost model
    - models/preprocessor.pkl: Preprocessing artifacts for inference
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import RANDOM_STATE, XGBOOST_PARAMS
from src.preprocessing import preprocess_training_data


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("TELCO CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # 1. Load data
    print("\n📂 Loading data...")
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    print(f"   ✓ Loaded {len(df)} records with {len(df.columns)} columns")
    
    # 2. Preprocess data
    print("\n🔧 Preprocessing data...")
    X, y, artifacts = preprocess_training_data(df)
    print(f"   ✓ Feature matrix shape: {X.shape}")
    print(f"   ✓ Target distribution - No Churn: {(y==0).sum()}, Churn: {(y==1).sum()}")
    
    # 3. Train-test split
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"   ✓ Training set: {len(X_train)} samples")
    print(f"   ✓ Test set: {len(X_test)} samples")
    
    # 4. Apply SMOTE for class imbalance
    print("\n⚖️ Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"   ✓ Resampled training set: {len(X_train_resampled)} samples")
    print(f"   ✓ New distribution - No Churn: {(y_train_resampled==0).sum()}, Churn: {(y_train_resampled==1).sum()}")
    
    # 5. Train XGBoost model
    print("\n🚀 Training XGBoost model...")
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train_resampled, y_train_resampled)
    print("   ✓ Model training complete")
    
    # 6. Evaluate model
    print("\n📈 Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"\n   ✓ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   ✓ Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   ✓ Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   ✓ F1-Score:  {f1:.4f}")
    print(f"   ✓ ROC-AUC:   {roc_auc:.4f}")
    
    print("\n" + "-" * 60)
    print("Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"   TN: {cm[0][0]:4d}  |  FP: {cm[0][1]:4d}")
    print(f"   FN: {cm[1][0]:4d}  |  TP: {cm[1][1]:4d}")
    
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # 7. Save model and artifacts
    print("\n💾 Saving model and preprocessing artifacts...")
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'model.pkl')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(artifacts, preprocessor_path)
    
    print(f"   ✓ Model saved to: {model_path}")
    print(f"   ✓ Preprocessor saved to: {preprocessor_path}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("\nTo start the API server, run:")
    print("   uvicorn app:app --reload")
    

if __name__ == "__main__":
    main()
