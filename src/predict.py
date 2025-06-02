# src/predict.py

"""
	Load the exported model and TF-IDF vectorizer to run inference on a single string or a batch of string
    •	
"""

import argparse
import joblib
import yaml
import tensorflow


def main