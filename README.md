# Prediction of stroke

### Author: Andrii Zhurba
The relevant project is located in src directory. 
Exploratory data analysis is in ExploratoryDataAnalysis.ipynb file,
while the modeling part is in Modeling.ipynb file.

## Problem definition:

**Problem**: According to the World Health Organization (WHO), stroke is the second leading cause of death globally, responsible for approximately 11% of total deaths. Early detection and intervention can significantly reduce the risk of fatal outcomes and improve the quality of life for individuals at risk.

**Solution**: Provide more attention to patients who have a high chance of experiencing a stroke. This may include prescribing appropriate medications, recommending lifestyle changes, and offering other preventive measures to help patients avoid strokes and improve their overall health.

**Machine Learning Solution**: Develop a predictive model that estimates the probability of a patient experiencing a stroke. By assessing individual risk levels, healthcare providers can prioritize care and implement preventive measures for high-risk patients. The model should output the likelihood of a stroke for each patient, enabling a data-driven approach to stroke prevention.

## Objective:

**Model Selection**: We will compare various machine 
learning models to identify the most effective one 
for our prediction task. We will assess the 
performance of different models using metrics 
such as accuracy, recall, precision, f1 and 
area under the curve of ROC curve. This 
involves tuning hyperparameters, moder
ensembling. To choose best model, we 
tried to focus on choosing the one with 
the highest ROC AUC. This helped in 
identifiying model with the high accuracy,
while keeping recall and precision on a good level. 

**Feature Importance**: Understanding which features 
have the most influence on the model’s predictions 
is crucial. We will analyze feature importance to 
determine the key drivers behind the target variable 
and evaluate how these features contribute to the 
model’s performance.

**Model deployment**: We deployed our best-performing model on our own computer, using it as a server. The Flask framework was used for deployment, and Docker was utilized to containerize the application.

## Results:

* **Best model**: **XGBoost** model has been chosen as our best model.
It achieved:
  * 73% accuracy
  * 76% recall
  * 12% precision
  * 21% f1-score 
  
  As this model has the highest ROC AUC, we can adjust 
  the threshold to balance the above metrics.
* **Feature importance**: **Age** has turned out to be the most
determining feature in our model. Almost all people,
who have had stroke, are older than 30 years old. 
BMI, average glucose level, hypertension, heart_disease,
work-type, smoking_status has also shown some importance
in determining the stroke. However, their influence is
tiny compared to age feature.

## Recommendations on solving the problem:
* Focus on Elderly Care: Elderly individuals are at the highest risk of stroke. Ensuring that this group has access to appropriate medication and healthcare services would be highly beneficial.
* Address Hypertension and Heart Disease: Special attention should be given to individuals with hypertension and heart disease. Additionally, targeted support is needed for those who are self-employed or work in government roles. Encouraging lifestyle improvements, promoting physical activity, and addressing mental health issues within these groups could significantly improve their well-being.

To run the notebook, install the following dependencies:

### Source:
The data was downloaded on kaggle website: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

```
pip install -r requirements.txt
```

## Building and Running the Docker Container
In the main directory of the project, build the docker
with the following command:
```bash
docker build -t stroke_prediction:latest -f deployment/Dockerfile .
```
Run the docker with the following command:
```bash
docker run -it --rm -p 8989:8989 stroke_prediction:latest
```



