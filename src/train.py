# src/train.py

"""
	•	Loads defaults from config.yaml
	•	Parses CLI flags with those defaults
	•	Performs data loading, oversampling, class weighting
	•	Builds & trains your MLP/LSTM via src.models.build_model
	•	Logs everything in MLflow
	•	Applies your recall gate
	•	Saves the final model
"""

import argparse  
import os
import yaml
import joblib
import mlflow
import numpy as np
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight # Intergrate to model.fit() so model pays more attention to the fake-job class
from imblearn.over_sampling import RandomOverSampler
from src.models import build_model

# Define and parse arguments for flexibility between MLP vs LSTM, tune batch size, epochs, dropout, thresholds, etc.
def parse_args(cfg):
    p = argparse.ArgumentParser(description="Train & evaluate fake-job detection")
    p.add_argument("--model_type", choices=["mlp", "lstm"],
                        default=cfg["training"]["model_type"],
                        help = "Which architecture to train")
    p.add_argument("--batch_size", type=int,
                        default=cfg["training"]["batch_size"],
                        help = "Number of samples per gradient update")
    p.add_argument("--epochs", type=int,
                        default=cfg["training"]["epochs"],
                        help = "Training epochs")
    p.add_argument("--dropout", type=float,
                        default=cfg["models"]["mlp"]["dropout"],
                        help = "Dropout rate for MLP")
    p.add_argument("--vocab_size", type=int,
                        default=cfg["models"]["lstm"]["vocab_size"],
                        help = "Vocabulary size (LSTM only)")
    p.add_argument("--embed_dim", type=int, 
                        default=cfg["models"]["lstm"]["embed_dim"],
                        help = "Embedding dimension (LSTM only)")
    p.add_argument("--lstm_units", type=int,
                        default=cfg["models"]["lstm"]["lstm_units"],
                        help = "Number of LSTM units")
    p.add_argument("--min_recall", type=float,
                        default=cfg["training"]["min_recall"],
                        help = "Minimum recall threshold for production")
    p.add_argument("--decision_threshold", type=float,
                        default=cfg["training"]["decision_threshold"],
                        help = "Threshold on predicted probability to call a posting 'fake'")
    return p.parse_args()


# Main training loop with MLflow -- wraps build, compile, train, evaluate, gate, and save all in one run that MKflow records
def main(config_path="src/config.yaml"):
    # 1. Loads config.yaml and parses the CLI arguments into Python object (args) and Sets up MLflow
    cfg = yaml.safe_load(open(config_path))
    args = parse_args(cfg) 
    mlflow.set_experiment("fake-job-detection")

    # 2. Loads data
    X_train, y_train = joblib.load(cfg["paths"]["train_split"])    # train_split: data/processed/train.pkl
    X_val,   y_val   = joblib.load(cfg["paths"]["val_split"])      # val_split: data/processed/val.pkl
    
    # 3. Oversamples minority class
    ros = RandomOverSampler(random_state=42)                # Duplicates samples from minority class until balanced, leaves majority class unchanged
    X_train, y_train = ros.fit_resample(X_train, y_train)
    
    # 4. Compute class weights to address imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", 
                                    classes=classes, 
                                    y=y_train)
    class_weights = dict(zip(classes, weights))

    # 5. Build model
    if args.model_type == "mlp":
         vec = joblib.load(cfg["paths"]["tfidf_path"])  # tfidf_path: data/processed/tfidf_vectorizer.joblib
         args.input_dim = len(vec.vocabulary_)          # Number of features : the size of the vocabulary learned by TF-IDF vectorizer which beccomes the input dimension for MLP model
    model = build_model(args)

    model.compile(
            optimizer="adam",               # Algorithm used to update weights during training
            loss="binary_crossentropy",     # measures how well the model's predictions match the actual labels
            metrics=["accuracy"]
        )
    
    # 6. Training & Logging - Start a new MLflow run, named after our model type
    with mlflow.start_run(run_name=args.model_type):
        mlflow.log_params(vars(args))   # converts CLI arguments args into dict (key: value) so they can be logged by mlflow
        mlflow.log_params({f"class_weight_{c}": w for c, w in class_weights.items()})
        
        # Train with class weights
        history = model.fit(
            X_train, 
            y_train,
            validation_data=(X_val,y_val),
            batch_size=args.batch_size,
            epochs=args.epochs,
            class_weight=class_weights,
            verbose=2
        )   

        # Predict probabilities & Compute metrics on validation
        predictions_probability = model.predict(X_val).ravel()                            # Keras model's predict() runs a forward pass over each batch in X_val and NumPy ravel() returns flattened copy of the array
        predictions = (predictions_probability >= args.decision_threshold).astype(int)    # Turn probabilities to 0/1 labels: (np.array([0.23,0.87,0.49]) > 0.5).astype(int) >>> array([0, 1, 0])

        recall    = recall_score(y_val, predictions)
        precision = precision_score(y_val, predictions)
        auc       = roc_auc_score(y_val, predictions_probability)

        mlflow.log_metrics({
            "val_recall": recall,
            "val_precision": precision,
            "val_auc": auc
        })
        print(f"Validation → Recall: {recall:.3f}, "
              f"Precision: {precision:.3f}, AUC: {auc:.3f}")
        
        if recall < args.min_recall:
            raise RuntimeError(f"Recall {recall:.3f} below threshold {args.min_recall}") # Set a hard gate to prevent accidental deployment of a weak model: if recall dips below our threshold, the script errors out
        
        # Save model to production folder
        out_dir = os.path.join(cfg["paths"]["production_dir"], args.model_type)
        os.makedirs(out_dir, exist_ok=True)
        model.export(out_dir)       # Export as a TF SavedModel

        mlflow.log_artifacts(out_dir,
                             artifact_path="production_models")
        
        print(f"✅ Model '{args.model_type}' exported to '{out_dir}' and logged to MLflow "
              f"With threshold {args.decision_threshold}")

if __name__ == "__main__":
        main()    