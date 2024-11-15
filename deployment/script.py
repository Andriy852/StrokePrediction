import requests
import time
from tqdm import tqdm
import numpy as np

url = "http://0.0.0.0:8989/predict"
application = {'gender': 'Male',
 'age': 53.0,
 'hypertension': 1,
 'heart_disease': 1,
 'ever_married': 'No',
 'work_type': 'Self-employed',
 'Residence_type': 'Urban',
 'avg_glucose_level': 104.55,
 'bmi': 30.9,
 'smoking_status': 'Smokes'}

all_times = []

for i in tqdm(range(1000)):
  t0 = time.time_ns() // 1_000_000

  resp = requests.post(url, json=application)
  t1 = time.time_ns() // 1_000_000

  time_taken = t1 - t0
  all_times.append(time_taken)

print("Response time in ms:")
print("Median:", np.quantile(all_times, 0.5))
print("95th precentile:", np.quantile(all_times, 0.95))
print("Max:", np.max(all_times))