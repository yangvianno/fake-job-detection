# src/train.py

"""
Created a fully parameterized, MLflow-tracked training pipeline that enforces a quality gate on recall before exporting the model. 
Everything is configurable via CLI flags, ensuring reproducibility and safety in CI/CD
"""

import argparse  
import os
import joblib
import mlflow
import numpy as np
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight # Intergrate to model.fit() so model pays more attention to the fake-job class
from imblearn.over_sampling import RandomOverSampler
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
    p.add_argument("--decision_threshold", type=float, default=0.50,
                        help = "Threshold on predicted probability to call a posting 'fake'")
    return p.parse_args()


# Load the processed train/validation splits by preprocess.py
def load_data():
    X_train, y_train = joblib.load("data/processed/train.pkl")
    X_val,   y_val   = joblib.load("data/processed/val.pkl")
    return X_train, y_train, X_val, y_val


# Main training loop with MLflow -- wraps build, compile, train, evaluate, gate, and save all in one run that MKflow records
def main():
    cfg = parse_args()      # Reads and perses the CLI arguments into Python object (cfg)
    mlflow.set_experiment("fake-job-detection")

    # === NEW: infer input_dim for MLP ===
    if cfg.model_type == "mlp":
        vec = joblib.load("data/processed/tfidf_vectorizer.joblib")
        cfg.input_dim = len(vec.vocabulary_)    # Number of features : the size of the vocabulary learned by TF-IDF vectorizer which beccomes the input dimension for MLP model
    
    # Start a new MLflow run, named after our model type
    with mlflow.start_run(run_name=cfg.model_type):
        # 1. Log hyperparameters
        mlflow.log_params(vars(cfg)) # converts CLI arguments cfg into dict (key: value) so they can be logged by mlflow

        # 2. Build and compile the model
        model = build_model(cfg)
        model.compile(
            optimizer="adam",               # Algorithm used to update weights during training
            loss="binary_crossentropy",     # measures how well the model's predictions match the actual labels
            metrics=["accuracy"]
        )

        # 3. Load data & Oversample minority class
        X_train, y_train, X_val, y_val = load_data()

        ros = RandomOverSampler(random_state=42)                # Duplicates samples from minority class until balanced, leaves majority class unchanged
        X_train, y_train = ros.fit_resample(X_train, y_train)
        mlflow.log_param("oversampled", True)

        # 4. Compute class weights to address imbalance
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", 
                                       classes=classes, 
                                       y=y_train)
        class_weights = dict(zip(classes, weights))
        mlflow.log_params({
            "class_weight_0" : class_weights.get(0),
            "class_weight_1" : class_weights.get(1),
        })
        
        # 5. Train with class weights
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val,y_val),
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            class_weight=class_weights,
            verbose=2
        )   

        # 6. Predict probabilities & Apply decision threshold for "fake" label
        predictions = model.predict(X_val).ravel()                            # Keras model's predict() runs a forward pass over each batch in X_val and NumPy ravel() returns flattened copy of the array

        y_prediction = (predictions >= cfg.decision_threshold).astype(int)    # Turn probabilities to 0/1 labels: (np.array([0.23,0.87,0.49]) > 0.5).astype(int) >>> array([0, 1, 0])


        # 7. Compute metrics on validation & Log evaluation metrics
        recall       = recall_score(y_val, y_prediction)
        precision    = precision_score(y_val, y_prediction)
        auc          = roc_auc_score(y_val, predictions)

        mlflow.log_metrics({
            "val_recall": recall,
            "val_precision": precision,
            "val_auc": auc
        })
        print(f"Validation → Recall: {recall:.3f}, "
              f"Precision: {precision:.3f}, AUC: {auc:.3f}")

        # 8. Quality gate
        if recall < cfg.min_recall:
            raise RuntimeError(f"Recall {recall:.3f} below threshold {cfg.min_recall}") # Set a hard gate to prevent accidental deployment of a weak model: if recall dips below our threshold, the script errors out
        
        # 9. Save to production folder & Log the saved model artifact -- after saving to local, now log to MLflow for experiment tracking, versioning, and preducibility
        out_dir = os.path.join("models/production", cfg.model_type)
        os.makedirs(out_dir, exist_ok=True)
        model.export(out_dir)       # Export as a TF SavedModel

        mlflow.log_artifacts(out_dir,
                             artifact_path=f"models/{cfg.model_type}")
        
        print(f"✅ Model '{cfg.model_type}' exported to '{out_dir}' and logged to MLflow"
              f"With threshold {cfg.decision_threshold}")

if __name__ == "__main__":
        main()    