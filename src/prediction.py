import sys
import os

print("Current working directory:", os.getcwd())
print("Python sys.path:")
for path in sys.path:
    print(path)

from flask import Flask, render_template, request
import numpy as np
import joblib
from preprocessing import preprocess_data
from model import load_model
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
parent_dir = os.path.dirname(current_dir)

# Explicitly define the template folder
app = Flask(__name__, template_folder=os.path.join(parent_dir, 'templates'))

# Load the trained model
model_path = 'models/model_diabetes.pkl'
loaded_model = load_model(model_path)

# Initialize the scaler
fitted_scaler = None

# Load the scaler
def load_fitted_scaler(data_path):
    global fitted_scaler
    train_data_path = os.path.join(parent_dir, 'data', 'train', 'diabetes.csv')
    X_processed, y_processed, fitted_scaler = preprocess_data(train_data_path)
    return fitted_scaler

load_fitted_scaler('')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    bloodpressure = float(request.form['bloodpressure'])
    skinthickness = float(request.form['skinthickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetespedigreefunction = float(request.form['diabetespedigreefunction'])
    age = float(request.form['age'])

    input_features = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]])

    # Scale the input features using the fitted scaler
    scaled_features = fitted_scaler.transform(input_features)

    # Make the prediction
    prediction = loaded_model.predict(scaled_features)

    # Interpret the prediction
    if prediction[0] == 1:
        result = "You are predicted to have diabetes."
    else:
        result = "You are predicted not to have diabetes."

    # Render the result template and pass the result
    return render_template('result.html', result=result)

if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    app.run(debug=True)