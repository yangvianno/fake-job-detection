# src/predict.py

"""
	This CLI will take one (or a batch of) raw job‐description(s), vectorize via TF-IDF, load the SavedModel, and output a probability + “fraudulent” label	
"""

import argparse
import joblib
import yaml
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TFSMLayer


def main(config_path="src/config.yaml", input_text=None):
    cfg = yaml.safe_load(open(config_path))

    vec = joblib.load(cfg["paths"]["tfidf_path"])           # Load TF-IDF vectorizer
    savedmodel_path = f"{cfg['paths']['production_dir']}/mlp" 
    model = Sequential([ TFSMLayer(savedmodel_path) ])      # Wrap the SavedModel via a TFSMLayer

    X = vec.transform([input_text])                         # Vectorize input
    X = tensorflow.cast(X.toarray(), tensorflow.float32)

    predict = float(model.predict(X).ravel()[0])
    label = int(predict >= cfg["training"]["decision_threshold"])
    print(f"Probability fake: {predict:.3f} → Label: {label}") # Because TFSMLayer expects float32, cast if needed

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True, help="Job description to classify")
    args = p.parse_args()
    main(input_text=args.text)
