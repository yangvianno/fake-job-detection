# src/models.py

import tensorflow

def build_mlp(input_dim: int, dropout_rate: float = 0.5) -> tensorflow.keras.Model:
    """Simple MLP on TF-IDF vectors""" # Multilayer Perceptron (MLP) - a type NN # Term Frequency--Inverse Document Frequency (TF-IDF)

    inputs = tensorflow.keras.Input(shape=(input_dim,), name="tfidf_input")    # type: ignore # reLU : Rectified Linear Unit
    x = tensorflow.keras.layers.Dense(64, activation='relu')(inputs)           # Adds first Dense (hidden) layer with 64 neurons and introduces non-linearity, healping the model learn complex patterns
    x = tensorflow.keras.layers.Dropout(dropout_rate)(x)  # 0.5 50% neurons dropped # Turn off some neurons during training to prevent overfitting
    x = tensorflow.keras.layers.Dense(32, activation='relu')(x)
    x = tensorflow.keras.layers.Dropout(dropout_rate)(x)               # 2nd Dense layer: Receive input from previous dropout layer

    outputs = tensorflow.keras.layers.Dense(1, activation="sigmoid", name="output")(x) # Final prediction: squashes output to a value between 0 & 1
    model   = tensorflow.keras.Model(inputs=inputs, outputs=outputs, name="mlp_model")

    return model

def build_lstm(vocab_size: int, embed_dim: int=128, lstm_units: int=64) -> tensorflow.keras.Model:      # 128: Dimensions of dense embdedding vectors, 64 neurons in LSTMM layer 
    """LSTM over token IDs (if later swap TF-IDF for tokenization)"""

    inputs = tensorflow.keras.Input(shape=(None,), dtype="int32", name="token_input")
    x = tensorflow.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inputs) # mask_zero tells layer to ignore padding assuming use 0 for padding in the sequences]
    x = tensorflow.keras.layers.LSTM(lstm_units)(x)     # Process sequence of embeddings through LSTM layer, ultimately returning the final vector for the LSTM

    outputs = tensorflow.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = tensorflow.keras.Model(inputs=inputs, outputs=outputs, name="lstm_model")

    return model

def build_model(cfg) -> tensorflow.keras.Model:
    """Dispatch based on cfg.model_type:
    1. 'mlp' expects cfg.input_dim    
    2. 'lstm' expects cfg.vocab_size"""

    if cfg.model_type == "mlp":
        return build_mlp(input_dim=cfg.input_dim, dropout_rate=cfg.dropout) # input_dim will be 5000 since they covered ~95% token occurrences
    elif cfg.model_type == "lstm": 
        return build_lstm(vocab_size=cfg.vocab_size,
                          embed_dim=cfg.embed_dim,
                          lstm_units=cfg.lstm_units)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")