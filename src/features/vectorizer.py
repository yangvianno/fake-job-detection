# src/features/vectorizer.py

import joblib
from pathlib import Path

class TfidfVectorizerWrapper:
    def __init__(self, path):
        self.path = Path(path)
    
    def load(self):
        return joblib.load(self.path)
    
    def transform(self, texts):
        vec = self.load()
        return vec.transform(texts)