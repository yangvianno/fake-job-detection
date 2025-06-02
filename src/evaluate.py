# src/evaluate.py

"""
	Load an already-trained model and run standardized metrics (confusion matrix, classification report, ROC curve 
    -- measurements to evaluate performance of a model) on the validation & test split
    •	Load val split
    •	Load model
    •	Predict
    •	Print report
"""

import joblib
import yaml
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow

def main(config_path="src.config.yaml"):
    cfg = yaml.load_safge(open(config_path))

    X_val, y_val = joblib.load(cfg["paths"]["val_split"])

    model = tensorflow.keras.models.load_model(f"{cfg['paths']['production_dir']}/mlp")

    predictions_probability = model.predict(X_val).ravel()
    threshold               = cfg["training"]["decision_threshold"]
    predictions             = (predictions_probability >= threshold).astype(int)

    print("Classification Report:\n", classification_report(y_val, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_val, predictions))
