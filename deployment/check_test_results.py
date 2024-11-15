import pandas as pd
import requests
import numpy as np

# load the data
test_set = pd.read_csv("../src/test_set.csv", index_col=0)
url = "http://127.0.0.1:8989/predict_testset"

# replace np.nan with None values
test_set.replace(np.nan, None, inplace=True)

# make a request
resp = requests.post(url, json=test_set.to_dict())

# print the results
print(resp.json())