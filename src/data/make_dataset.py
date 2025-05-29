# src/data/make_dataset.py

import os
import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from ..preprocess import preprocess_df

def make_dataset(config_path="src/config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    raw = pd.read_csv(cfg["path"]["raw_csv"])
    clean = preprocess_df(raw)

    vec = TfidfVectorizer(
        max_features=cfg["models"]["lstm"]["vocab_size"],
        ngram_range=(1,2),
        stop_words="english"
    )

    X = vec.fit_transform(clean["description"])
    os.makedirs(cfg["paths"]["processed_dir"], exist_ok=True)
    joblib.dump(vec, cfg["paths"]["tfidf_path"])

    n = int(0.8 * X.shape[0])
    train = (X[:n], clean["fraudulent"][:n])
    val   = (X[n:], clean["fraudulent"][n:])
    joblib.dump(train, cfg["paths"]["train_split"])
    joblib.dump(val,   cfg["paths"]["val_split"])
    print("Data prepared 🎉")