import sys
import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib
from preprocessing import preprocess_data
from model import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
parent_dir = os.path.dirname(current_dir)

# Explicitly define the template folder and static folder
app = Flask (__name__, template_folder=os.path.join(parent_dir, 'templates'),
static_folder=os.path.join(parent_dir, 'static'))

# Define the directory to save uploaded files
UPLOAD_FOLDER = os.path.join(parent_dir, 'data', 'uploaded')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model_path = os.path.join(parent_dir, 'models', 'model_diabetes.pkl')
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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('retrain_status.html', message='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('retrain_status.html', message='No selected file')
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            return render_template('retrain_status.html', message=f'File "{filename}" uploaded successfully.')
        except Exception as e:
            return render_template('retrain_status.html', message=f'Error uploading file: {e}')
    return render_template('retrain_status.html', message='Something went wrong')

@app.route('/retrain')
def retrain_model():
    print("Retraining initiated!")
    uploaded_data_folder = app.config['UPLOAD_FOLDER']
    train_data_path = os.path.join(parent_dir, 'data', 'train', 'diabetes.csv')
    model_path = os.path.join(parent_dir, 'models', 'model_diabetes.pkl')

    try:
        # Load the original training data
        train_df = pd.read_csv(train_data_path)
        X_train_original = train_df.drop('Outcome', axis=1)
        y_train_original = train_df['Outcome']

        # Load uploaded data (if any)
        uploaded_files = [f for f in os.listdir(uploaded_data_folder) if f.endswith('.csv')]
        num_uploaded_files = len(uploaded_files)
        uploaded_dfs = []
        for file in uploaded_files:
            file_path = os.path.join(uploaded_data_folder, file)
            try:
                df = pd.read_csv(file_path)
                if 'Outcome' not in df.columns:
                    return render_template('retrain_status.html', message=f"Error: Uploaded file '{file}' is missing the 'Outcome' column.")
                uploaded_dfs.append(df)
            except Exception as e:
                return render_template('retrain_status.html', message=f"Error reading uploaded file {file}: {e}")

        if uploaded_dfs:
            # Concatenate all uploaded dataframes
            uploaded_df = pd.concat(uploaded_dfs, ignore_index=True)
            # Assuming the uploaded data has the same structure as the training data
            X_uploaded = uploaded_df.drop('Outcome', axis=1)
            y_uploaded = uploaded_df['Outcome']

            # Combine original training data with uploaded data
            X_combined = pd.concat([X_train_original, X_uploaded], ignore_index=True)
            y_combined = pd.concat([y_train_original, y_uploaded], ignore_index=True)
        else:
            X_combined = X_train_original
            y_combined = y_train_original

        # Combine features and target into a single DataFrame
        combined_df = pd.concat([X_combined, y_combined], axis=1)

        # Preprocess the combined data
        X_processed, y_processed, fitted_scaler = preprocess_data(combined_df) # Pass the DataFrame directly

        # Train a new Logistic Regression model
        new_model = LogisticRegression(solver='liblinear', random_state=42) # Using the same model as likely before
        new_model.fit(X_processed, y_processed)

        # Save the retrained model
        joblib.dump(new_model, model_path)

        # Reload the fitted scaler
        load_fitted_scaler('') # This will refit the scaler on the new training data for future predictions

        if num_uploaded_files > 0:
            return render_template('retrain_status.html', message=f"Model retrained successfully using {num_uploaded_files} new file(s)!")
        else:
            return render_template('retrain_status.html', message="Model retrained successfully using existing data.")

    except Exception as e:
        return render_template('retrain_status.html', message=f"Error during retraining: {e}")

if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    app.run(debug=True)