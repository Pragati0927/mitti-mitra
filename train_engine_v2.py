"""
Mitti Mitra - Updated Training Pipeline
Trains a multi-class crop recommendation model on the expanded dataset.

Features used (Phase 1 compatible):
- past_crop
- latitude, longitude
- temperature_c, humidity_pct, rainfall_mm
- inferred soil nutrients (N, P, K, pH, OM)

Outputs model assets that prediction_engine.py can load.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_PATH = "master_indian_recommendation_dataset_expanded.csv"
MODEL_DIR = Path("model_assets")
MODEL_DIR.mkdir(exist_ok=True)


def train():
    logging.info("Loading expanded dataset...")
    df = pd.read_csv(DATA_PATH)
    logging.info(f"Loaded {len(df)} rows")

    # Features
    categorical = ["past_crop"]
    numeric = [
        "latitude", "longitude",
        "temperature_c", "humidity_pct", "rainfall_mm",
        "inferred_N_ppm", "inferred_P_ppm", "inferred_K_ppm",
        "inferred_pH", "inferred_OM_pct",
    ]
    target = "recommended_crop"

    # Encode categorical
    label_encoders = {}
    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Target encoder
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[target])

    X = df[categorical + numeric].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    logging.info("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    logging.info(f"Validation Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        y_val, y_pred,
        target_names=target_encoder.classes_,
        zero_division=0
    ))

    # Save assets
    with open(MODEL_DIR / "xgb_crop_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "target_encoder.pkl", "wb") as f:
        pickle.dump(target_encoder, f)
    with open(MODEL_DIR / "label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

    # Also save feature order for inference
    with open(MODEL_DIR / "feature_order.pkl", "wb") as f:
        pickle.dump(categorical + numeric, f)

    logging.info(f"Model assets saved to {MODEL_DIR}/")
    return model, acc


if __name__ == "__main__":
    train()
