import requests

# list of object type features
obj_features = ["gender", "ever_married",
                "work_type", "Residence_type", "smoking_status"]

# list of numeric features
num_features = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]

data = {}

# possible values for object type features
feature_values = {"ever_married": ["Yes", "No"],
                  "work_type": ["Self-employed", "Private", "Govt_job", "children"],
                  "Residence_type": ["Urban", "Rural"],
                  "smoking_status": ['formerly smoked', 'never smoked', 'smokes', 'Unknown'],
                  "gender": ["Male", "Female"]}

# add object features
for feature in obj_features:
 data[feature] = input(f"Enter the value of {feature}({'/'.join(feature_values[feature])}): ")

# add numeric features
for feature in num_features:
 data[feature] = float(input(f"Enter the value of {feature}: "))

# make a request
url = "http://0.0.0.0:8989/predict"
resp = requests.post(url, json=data)
print(resp.json())