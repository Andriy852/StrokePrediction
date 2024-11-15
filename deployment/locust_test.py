from locust import HttpUser, task, constant_throughput

test_application = {'gender': 'Male',
 'age': 53.0,
 'hypertension': 1,
 'heart_disease': 1,
 'ever_married': 'No',
 'work_type': 'Self-employed',
 'Residence_type': 'Urban',
 'avg_glucose_level': 104.55,
 'bmi': 30.9,
 'smoking_status': 'Smokes'}

class StrokePrediction(HttpUser):
    # 1 request per second
    wait_time = constant_throughput(1)

    @task
    def predict(self):
        self.client.post("/predict", json=test_application, timeout=1)