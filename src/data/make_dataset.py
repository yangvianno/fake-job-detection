# src/data/make_dataset.py

import os
import joblib
import pandas as pd
import yaml
from src.preprocess import preprocess_df
from src.features.vectorizer import TfidfVectorizerWrapper

def make_dataset(config_path: str = "src/config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    paths = cfg["paths"]

    df = pd.read_csv(paths["raw_csv"])  # raw = pd.read_csv(cfg["path"]["raw_csv"])
    df_clean = preprocess_df(df)

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(
        max_features=cfg["models"]["lstm"]["vocab_size"],
        ngram_range=(1, 2),
        stop_words="english"
    )

    X = vec.fit_transform(df_clean["description"])
    os.makedirs(paths["processed_dir"], exist_ok=True)
    joblib.dump(vec, paths["tfidf_path"])

    n_train = int(cfg["training"]["train_ratio"] * X.shape[0])  # src/config.yaml
    X_train = X[:n_train]
    X_val   = X[n_train:]
    y_train = df_clean["fraudulent"][:n_train]
    y_val   = df_clean["fraudulent"][n_train:]

    joblib.dump((X_train, y_train), paths["train_split"])
    joblib.dump((X_val,   y_val),   paths["val_split"])
    print(f"✅ Dataset ready — train: {n_train}, val: {X.shape[0] - n_train}")

if __name__ == "__main__":
    make_dataset()