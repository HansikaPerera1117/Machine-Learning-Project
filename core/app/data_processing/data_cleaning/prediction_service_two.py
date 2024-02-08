import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from app.data_processing.data_cleaning.prediction_service import get_user_input
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

dataset = pd.read_csv(
    r"F:\ijse final\ai-module-backend-2\core\csv\Heart_disease_prediction.csv",
    header=None,
)

# Set column names
dataset.columns = [
    "age",
    "sex",
    "cp",
    "restbp",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "hd",
]

# Handle missing values
df_no_missing = dataset.loc[(dataset["ca"] != "?") & (dataset["thal"] != "?")]

# Prepare features and target variable
X = df_no_missing.drop("hd", axis=1).copy()
y = df_no_missing["hd"].copy()

# One-Hot Encode categorical features
X_encoded = pd.get_dummies(X, columns=["cp", "restecg", "slope", "thal", "oldpeak"])

# Binary classification: 0 for no heart disease, 1 for heart disease
y_not_zero_index = y > 0
y[y_not_zero_index] = 1

#  methin iwiri data cleain


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)

# Train a decision tree classifier with cost-complexity pruning
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)

# Perform cost-complexity pruning and plot accuracy vs alpha
path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
ccp_alphas = ccp_alphas[:-1]

clf_dts = []
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)

# Plot accuracy vs alpha for training and testing sets
train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
# plt.show()

# Choose an optimal alpha and create a pruned decision tree
clf_dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016)
clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

# Plot the pruned decision tree
plt.figure(figsize=(15, 7.5))
plot_tree(
    clf_dt_pruned,
    filled=True,
    rounded=True,
    class_names=["No HD", "Yes HD"],
    feature_names=X_encoded.columns,
)


def get_user_datas(data):
    user_data = {
        "age": data.age,
        "sex": data.sex,
        "cp": data.cp,
        "restbp": data.restbp,
        "chol": data.chol,
        "fbs": data.fbs,
        "restecg": data.restecg,
        "thalach": data.thalach,
        "exang": data.exang,
        "oldpeak": data.oldpeak,
        "slope": data.slope,
        "ca": data.ca,
        "thal": data.thal,
    }
    print(user_data)
    user_data_df = pd.DataFrame([user_data])
    user_data_encoded = pd.get_dummies(
        user_data_df, columns=["cp", "restecg", "slope", "thal"]
    )
    user_data_encoded = user_data_encoded.reindex(
        columns=X_encoded.columns, fill_value=0
    )
    prediction = clf_dt_pruned.predict(user_data_encoded)
    probability = clf_dt_pruned.predict_proba(user_data_encoded)[:, 1]
    if prediction[0] == 1:
        risk_percentage = probability[0] * 100
        print(
            f"The model predicts that the person has heart disease with a risk of {risk_percentage:.2f}%."
        )
        person_tag = "High Risk" if risk_percentage > 50 else "Low Risk"
        print("Person's Tag:", person_tag)
        return {person_tag, risk_percentage}
    else:
        risk_percentage = (1 - probability[0]) * 100
        print(
            f"The model predicts that the person does not have heart disease with a risk of {risk_percentage:.2f}%."
        )
        person_tag = "Low Risk" if risk_percentage > 50 else "High Risk"
        print("Person's Tag:", person_tag)
        return {person_tag, risk_percentage}
# 

def get_prediction(data):
    return get_user_datas(data)
