# Diabetes_Prediction
 A ml model based on random forest to predict diabetes is present in an individual or not.
 https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset, I have used this dataset for the model.
 Have added comments in the project to help you understand it one step at a time


```
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
dataset = pd.read_csv('D:\diabetes_prediction_dataset.csv')
x=dataset.iloc[:,:-1]
print(x)

data = pd.read_csv("D:\diabetes_prediction_dataset.csv")  # Replace this with the actual path to your CSV file

categorical_features = ["gender", "hypertension", "heart_disease", "smoking_history"]
for feature in categorical_features:
    data[feature] = LabelEncoder().fit_transform(data[feature])




numerical_features = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])


features = data.drop("diabetes", axis=1)
target = data["diabetes"]


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))


new_data = pd.DataFrame({
    "gender": ["Female"],
    "age": [50],
    "hypertension": [1],
    "heart_disease": [0],
    "smoking_history": ["never"],
    "bmi": [25],
    "HbA1c_level": [6.0],
    "blood_glucose_level": [120]
})


for feature in categorical_features:
    new_data[feature] = LabelEncoder().fit_transform(new_data[feature])

new_prediction = model.predict(new_data)
print("Predicted diabetes for new data:", new_prediction[0])```





