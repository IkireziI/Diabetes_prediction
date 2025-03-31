import joblib

def load_model(model_path):
    """Loads the trained model."""
    return joblib.load(model_path)

if __name__ == '__main__':
    # Example usage to test if the model loads (optional)
    model = load_model('../../models/model_diabetes.pkl')
    print("Model loaded successfully:", model)