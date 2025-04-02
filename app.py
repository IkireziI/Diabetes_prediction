
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the trained Keras model
model_path = 'models/model_diabetes_dropout_es.keras'
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    model = None

# Load the scaler
scaler_path = '/content/scaler.joblib'
try:
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded successfully from {scaler_path}")
except Exception as e:
    print(f"Error loading scaler from {scaler_path}: {e}")
    scaler = None

def predict_diabetes(features):
    """
    Predicts the probability of diabetes based on the input features.
    Assumes the input features are in the same order as used during training.
    """
    if model is None or scaler is None:
        return "Model or scaler not loaded."

    # Convert input features to a NumPy array
    feature_array = np.array(features).reshape(1, -1)

    # Scale the input features using the loaded scaler
    scaled_features = scaler.transform(feature_array)

    # Make prediction
    prediction_proba = model.predict(scaled_features)[0][0]

    # Apply threshold (0.5) to get binary prediction
    prediction = 1 if prediction_proba >= 0.5 else 0

    return prediction, prediction_proba

if __name__ == '__main__':
    # Example usage: Replace with your own feature values
    sample_features = [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0] # Example features
    if model is not None and scaler is not None:
        prediction, probability = predict_diabetes(sample_features)
        print(f"Sample features: {sample_features}")
        print(f"Predicted class (0: No Diabetes, 1: Diabetes): {prediction}")
        print(f"Predicted probability: {probability:.4f}")
    else:
        print("Cannot run example as model or scaler was not loaded.")
