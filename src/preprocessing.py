import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import os

scaler = MinMaxScaler()

def preprocess_data(data):
    """
    Preprocesses the diabetes dataset.
    Accepts either a file path or a pandas DataFrame.
    """
    if isinstance(data, str):
        dataset = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        dataset = data.copy() # Work with a copy to avoid modifying the original DataFrame
    else:
        raise ValueError("Input must be a file path (str) or a pandas DataFrame.")

    dataset[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan)
    # Impute NaN values (using median for robustness)
    dataset['Glucose'] = dataset['Glucose'].fillna(dataset['Glucose'].median())
    dataset['BloodPressure'] = dataset['BloodPressure'].fillna(dataset['BloodPressure'].median())
    dataset['SkinThickness'] = dataset['SkinThickness'].fillna(dataset['SkinThickness'].median())
    dataset['Insulin'] = dataset['Insulin'].fillna(dataset['Insulin'].median())
    dataset['BMI'] = dataset['BMI'].fillna(dataset['BMI'].median())

    # --- Handling Outliers (for demonstration, you might want to handle this differently in production) ---
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df_filtered

    dataset = remove_outliers_iqr(dataset, 'Insulin')

    # --- Handling Class Imbalance (for demonstration, you might want to handle this differently in prediction) ---
    smote = SMOTE(random_state=42)
    X = dataset.drop('Outcome', axis=1)
    y = dataset['Outcome']
    X_resampled, y_resampled = smote.fit_resample(X, y)
    dataset_resampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)

    # --- Feature Scaling ---
    X_scaled = scaler.fit_transform(dataset_resampled.drop('Outcome', axis=1))
    return X_scaled, dataset_resampled['Outcome'], scaler # Return the scaler as well

if __name__ == '__main__':
    # Example usage
    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(script_dir, '..', '..', 'data', 'train', 'diabetes.csv')
    X_processed, y_processed, fitted_scaler = preprocess_data(dataset_path)
    print("Preprocessing done. Shape of processed data:", X_processed.shape)