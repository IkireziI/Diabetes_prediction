import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def handle_missing_zeros(df, columns_to_replace):
    """Replaces zero values in specified columns with NaN."""
    df[columns_to_replace] = df[columns_to_replace].replace(0, pd.NA)
    return df

def impute_missing_values(df, strategy='mean'):
    """Imputes missing values using the specified strategy."""
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

def scale_features(df, numerical_features):
    """Scales numerical features using StandardScaler."""
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Splits the data into training and testing sets."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test