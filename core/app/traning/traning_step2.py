# import os

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from pylab import rcParams
# from sklearn import metrics
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
# from sklearn.preprocessing import StandardScaler

# lr = LinearRegression()
# lr.fit(X_train, Y_train)

# LinearRegression()

# print("LR Coefficients: \n", lr.coef_)
# print("LR Intercept: \n", lr.intercept_)


# print("Performance (R^2): ", lr.score(X_train, Y_train))


# print("Performance (R^2): ", lr.score(X_train, Y_train))


# def get_mape(y_true, y_pred):
#     """
#     Compute mean absolute percentage error (MAPE)
#     """
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Y_train_pred = lr.predict(X_train)
# Y_val_pred = lr.predict(X_val)
# Y_test_pred = lr.predict(X_test)


# df_pred = pd.DataFrame(Y_val.values, columns=["Actual"], index=Y_val.index)
# df_pred["Predicted"] = Y_val_pred
# df_pred = df_pred.reset_index()
# df_pred.loc[:, "Date"] = pd.to_datetime(df_pred["Date"], format="%Y-%m-%d")
# df_pred


# df_pred[["Actual", "Predicted"]].plot()
