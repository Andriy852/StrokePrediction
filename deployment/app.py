import pandas as pd
from flask import Flask, jsonify, request
import pickle
from utils import helper_functions

# load the model
with open("stroke_predictor.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask("default")

# function which will return the results on the test data
@app.route("/predict_testset", methods=["POST"])
def get_test_scores():
    data = request.get_json()
    data = pd.DataFrame(data)
    x_test = data.drop(columns=["id", "stroke"])
    true_y = data.loc[:, "stroke"]
    res = helper_functions.get_scores(model, x_test, true_y, fit=False)
    return jsonify(res)

# function which accepts the data about the patient and returns the chance of stroke
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    data = pd.DataFrame(data, index=[0])
    proba = model.predict_proba(data)[0, 1].tolist()
    return jsonify(proba)

if __name__ == "__main__":
    # Run the app on local host and port 8989
    app.run(debug=True, host="0.0.0.0", port=8989)


