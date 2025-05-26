#  src/api.py

"""
1. Load the exported SavedModel
2. Load the TF-IDF vectorizer
3. Define a Pydantic schema for the job post - a class used with FastAPI that defines and validates structured data
4. Expose a /predict endpoint that returns the fake-job probability
"""

import joblib
import numpy as np
import tensorflow as tf         # load and use the saved ML model
from fastapi import FastAPI     
from pydantic import BaseModel  # defines and validates the shape of incoming JSON data     

# 1. Load artifacts at start up
VEC_PATH   = "data/processed/tfidf_vectorizer.joblib"
MODEL_PATH = "models/production/mlp"        # Adjust if using LSTM later

vectorizer = joblib.load(VEC_PATH)
model      = tf.saved_model.load(MODEL_PATH)

# 2. Define request schema
class JobPost(BaseModel):
    title: str = ""
    description : str

# 3. Instantiate FastAPI App
app = FastAPI(title="Fake Job Detection API")

@app.post("/predict")
def predict(post: JobPost):
    # 4. Preprocess & vectorize
    full_text = (post.title + " " + post.description).strip()
    X = vectorizer.transform([full_text]).toarray().astype(np.float32)      # Applies the saved TfidfVectorizer to the input, Converts raw text into a TF-IDF feature vector

    # 5. Call the SavedModel 'serve' signature -- TF auto stores a "serve" signature (how to call the model) 
    infer = model.signatures["serve"]       # Loads the model's inference (serve) signture -- like the entry point to run predictions
    probs = infer(tf.constant(X))["output_0"].numpy().ravel() # converts dense NumPy array for TensorFlow by wrapping tf.constant into tensor
    
    return {"fake_probability": float(probs[0])}

@app.get("/")
def root():
    return {"message": "Fake Job Detection API is up. POST to /predict"}