# src/train.py

import argparse  
import os
import joblib
import mlflow
import numpy as np
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from src.model import build_model

# Define and parse arguments for flexibility between MLP vs LSTM, tune batch size, epochs, dropout, thresholds, etc.
def parse_args():
    p = argparse.ArgumentParser(description="Train & evaluate fake-job detection")
    p.add_argument("--model_type", choices=["mlp", "lstm"], required=True,
                        help = "Which architecture to train")
    p.add_argument("--batch_size", type=int, default=32,
                        help = "Number of samples per gradient update")
    p.add_argument("--epochs", type=int, default=5,
                        help = "Training epochs")
    p.add_argument("--dropout", type=float, default=0.5,
                        help = "Dropout rate for MLP")
    p.add_argument("--vocab_size", type=int, default=10000,
                        help = "Vocabulary size (LSTM only)")
    p.add_argument("--embed_dim", type=int, default=128, 
                        help = "Embedding dimension (LSTM only)")
    p.add_argument("--lstm_units", type=int, default=64,
                        help = "Number of LSTM units")
    p.add_argument("--min_recall", type=float, default=0.85,
                        help = "Minimum recall threshold for production")
    return p.parse_args()


# Load the processed train/validation splits by preprocess.py
def load_data():
    X_train, y_train = joblib.load("data/processed/train.pkl")
    X_val,   y_val   = joblib.load("data/processed/val.pkl")
    return X_train, y_train, X_val, y_val


# Main training loop with MLflow -- wraps build, compile, train, evaluate, gate, and save all in one run that MKflow records
def main():
    cfg = parse_args()
    mlflow.set_experiment("fake-job-detection")

    # Start a new MLflow run, named after our model type
    with mlflow.start_run(run_name=cfg.model_type):
        # 1. Log hyperparameters
        mlflow.log_params(vars(cfg)) # converts CLI arguments cfg into dict (key: value) so they can be logged by mlflow

        # === NEW: infer input_dim for MLP ===
    if cfg.model_type == "mlp":
        vec = joblib.load("data/processed/tfidf_vectorizer.joblib")
        cfg.input_dim = len(vec.vocabulary_)

        # 2. Build and compile the model
        model = build_model(cfg)
        model.compile(
            optimizer="adam",               # Algorithm used to update weights during training
            loss="binary_crossentropy",    # measures how well the model's predictions match the actual labels
            metrics=["accuracy"]
        )

        # 3. Load data
        X_train, y_train, X_val, y_val = load_data()

        # 4. Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val,y_val),
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            verbose=2
        )   

        # 5. Predict & Compute metrics
        predictions = model.predict(X_val).ravel()          # Keras model's predict() runs a forward pass over each batch in X_val and NumPy ravel() returns flattened copy of the array
        y_prediction = (predictions > 0.5).astype(int)      # Turn probabilities to 0/1 labels: (np.array([0.23,0.87,0.49]) > 0.5).astype(int) >>> array([0, 1, 0])
        recall       = recall_score(y_val, y_prediction)
        precision    = precision_score(y_val, y_prediction)
        auc          = roc_auc_score(y_val, predictions)

        # 6. Log metrics
        mlflow.log_metrics({
            "val_recall": recall,
            "val_precision": precision,
            "val_auc": auc
        })
        print(f"Validation → Recall: {recall:.3f}, Precision: {precision:.3f}, AUC: {auc:.3f}")

        # 7. Quality gate
        if recall < cfg.min_recall:
            raise RuntimeError(f"Recall {recall:.3f} below threshold {cfg.min_recall}") # Set a hard gate to prevent accidental deployment of a weak model: if recall dips below our threshold, the script errors out
        
        # 8. Save to production folder
        out_dir = os.path.join("models/production", cfg.model_type)
        os.makedirs(out_dir, exist_ok=True)
        model.save(out_dir)

        # 9. Log the saved model artifact
        mlflow.keras.log_model(model, artifact_path=f"models/{cfg.model_type}")

        print(f"✅ Model '{cfg.model_type}' saved to '{out_dir}'")

if __name__ == "__main__":
        main()    