from locust import HttpUser, task, between

class DiabetesPredictorUser(HttpUser):
    wait_time = between(1, 3) # Wait between 1 and 3 seconds between tasks

    @task
    def predict_diabetes(self):
        # Here we would define the data to send to our prediction endpoint
        payload = {
            "features": [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0] # Example features
        }
        self.client.post("/predict", json=payload)