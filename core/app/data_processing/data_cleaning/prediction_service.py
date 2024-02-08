import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# Function to get user input for prediction
def get_user_input(data):
    # age = int(input("Enter age: "))
    # sex = int(input("Enter sex (0 for female, 1 for male): "))
    # cp = int(input("Enter chest pain type (0-3): "))
    # restbp = int(input("Enter resting blood pressure: "))
    # chol = int(input("Enter cholesterol level: "))
    # fbs = int(input("Enter fasting blood sugar (0 for false, 1 for true): "))
    # restecg = int(input("Enter resting electrocardiographic results (0-2): "))
    # thalach = int(input("Enter maximum heart rate achieved: "))
    # exang = int(input("Enter exercise-induced angina (0 for no, 1 for yes): "))
    # oldpeak = float(input("Enter ST depression induced by exercise relative to rest: "))
    # slope = int(input("Enter the slope of the peak exercise ST segment (0-2): "))
    # ca = int(input("Enter number of major vessels colored by fluoroscopy (0-3): "))
    # thal = int(input("Enter thalassemia type (1-3): "))

    # user_data = {
    #     "age": data.age,
    #     "sex": data.sex,
    #     "cp": data.cp,
    #     "restbp": data.restbp,
    #     "chol": data.chol,
    #     "fbs": data.fbs,
    #     "restecg": data.restecg,
    #     "thalach": data.thalach,
    #     "exang": data.exang,
    #     "oldpeak": data.oldpeak,
    #     "slope": data.slope,
    #     "ca": data.ca,
    #     "thal": data.thal,
    # }

    user_data = {
        "age": 50.0,
        "sex": 1.0,
        "cp": 1.0,
        "restbp": 100.0,
        "chol": 100.0,
        "fbs": 0.0,
        "restecg": 1.0,
        "thalach": 100.0,
        "exang": 1.0,
        "oldpeak": 1.6,
        "slope": 2.0,
        "ca": 1.0,
        "thal": 2.0,
    }
    return user_data


# user_data = {
#     "age": 67.0,
#     "sex": 1.0,
#     "cp": 4.0,
#     "restbp": 120.0,
#     "chol": 229.0,
#     "fbs": 0.0,
#     "restecg": 2.0,
#     "thalach": 129.0,
#     "exang": 1.0,
#     "oldpeak": 2.6,
#     "slope": 2.0,
#     "ca": 2.0,
#     "thal": 7.0,
# }
# user_data = {
#     "age": 50.0,
#     "sex": 1.0,
#     "cp": 1.0,
#     "restbp": 100.0,
#     "chol": 100.0,
#     "fbs": 0.0,
#     "restecg": 1.0,
#     "thalach": 100.0,
#     "exang": 1.0,
#     "oldpeak": 1.6,
#     "slope": 2.0,
#     "ca": 1.0,
#     "thal": 2.0,
# }
