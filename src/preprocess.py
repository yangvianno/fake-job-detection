# src/preprocess.py

import os
import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

RAW_PATH = "data/raw/fake_job_postings.csv"
PROC_DIR = "data/processed"

def load_raw():
    return pd.read_csv(RAW_PATH)

def clean_text(text: str) -> str:
    """Remove simple PII tokens and placeholder tokens"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"#URL\\S*", "", text)
    text = re.sub(r"#EMAIL\\S*", "", text)
    text = re.sub(r"#PHONE\\S*", "", text)

    text = re.sub(r"https?://\\S+|www\\.\\S+", "", text)
    text = re.sub(r"\\S+@\\S+", "", text)
    text = re.sub(r"\\b\\d{3}[-\\.\\s]?\\d{3}[-\\.\\s]?\\d{4}\\b", "", text)
    return text.strip()

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to the job description field."""
    df = df.copy()      # to avoid modifying the original df in-place
    df["description"] = df["description"].fillna("").apply(clean_text) # a pandas method that applies a function to each element (row)
    return df

def vectorize_and_split(df: pd.DataFrame, max_features=5000):   # Limits top 5000 most frequent words
    """
    1. Transform text data into numerical features using TF-IDF vectorization
    2. Split the dataset into training and validation sets
    3. Saves:
        - tfidf_vectorizer.joblib
        - train.pkl, val.pkl (Tuples of (X,y))
    """
    os.makedirs(PROC_DIR, exist_ok=True)

    # 1. Vectorize description
    vec = TfidfVectorizer(max_features=max_features)
    X = vec.fit_transform(df["description"])    # Learns the vocab then transforms into numerical features

    # 2. Save the trained vertorizer to a file for later use inference/retraining
    joblib.dump(vec, os.path.join(PROC_DIR, "tfidf_vectorizer.joblib"))

    # 3. Split 80/20 train/validation sets
    n_train = int( 0.8 * X.shape[0])            # Ex: X.shape[0] = 1000 job postings
    X_train, X_val = X[:n_train], X[n_train:]   # TF-IDF features
    y_train = df["fraudulent"][:n_train]         # Binary fraud labels
    y_val = df["fraudulent"][n_train:]

    # 4. Save sets
    joblib.dump((X_train, y_train), os.path.join(PROC_DIR, "train.pkl"))
    joblib.dump((X_val, y_val), os.path.join(PROC_DIR, "val.pkl"))

def main():
    """
    Orchestrate: Load raw data -> Clean text -> Vectorize & Split
    Usage:
        python src/preprocess.py"""
    
    print("🛠  Starting preprocessing...")
    df = load_raw()
    print(f"✔️ Loaded raw data: {df.shape[0]} rows")

    df_clean = preprocess_df(df)
    print("✔️ Cleaning complete")

    vectorize_and_split(df_clean)
    print(f"✅ Preprocessing done. Files written to '{PROC_DIR}'")

if __name__ == "__main__":
    main()
