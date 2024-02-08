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


# def create_train_test_set(df_Stock):
#     features = df_Stock.drop(columns=["Close_forcast"], axis=1)
#     target = df_Stock["Close_forcast"]

#     data_len = df_Stock.shape[0]
#     print("Historical Stock Data length is - ", str(data_len))

#     # create a chronological split for train and testing
#     train_split = int(data_len * 0.88)
#     print("Training Set length - ", str(train_split))

#     val_split = train_split + int(data_len * 0.1)
#     print("Validation Set length - ", str(int(data_len * 0.1)))

#     print("Test Set length - ", str(int(data_len * 0.02)))

#     # Splitting features and target into train, validation and test samples
#     X_train, X_val, X_test = (
#         features[:train_split],
#         features[train_split:val_split],
#         features[val_split:],
#     )
#     Y_train, Y_val, Y_test = (
#         target[:train_split],
#         target[train_split:val_split],
#         target[val_split:],
#     )

#     # print shape of samples
#     print(X_train.shape, X_val.shape, X_test.shape)
#     print(Y_train.shape, Y_val.shape, Y_test.shape)

#     return X_train, X_val, X_test, Y_train, Y_val, Y_test


# X_train, X_val, X_test, Y_train, Y_val, Y_test = create_train_test_set(df_Stock)
# # Historical Stock Data length is -  3732
# # Training Set length -  3284
# # Validation Set length -  373
# # Test Set length -  74
# # (3284, 61) (373, 61) (75, 61)
# # (3284,) (373,) (75,)

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

# print("Training R-squared: ", round(metrics.r2_score(Y_train, Y_train_pred), 2))
# print(
#     "Training Explained Variation: ",
#     round(metrics.explained_variance_score(Y_train, Y_train_pred), 2),
# )
# print("Training MAPE:", round(get_mape(Y_train, Y_train_pred), 2))
# print(
#     "Training Mean Squared Error:",
#     round(metrics.mean_squared_error(Y_train, Y_train_pred), 2),
# )
# print(
#     "Training RMSE: ",
#     round(np.sqrt(metrics.mean_squared_error(Y_train, Y_train_pred)), 2),
# )
# print("Training MAE: ", round(metrics.mean_absolute_error(Y_train, Y_train_pred), 2))

# print(" ")

# print("Validation R-squared: ", round(metrics.r2_score(Y_val, Y_val_pred), 2))
# print(
#     "Validation Explained Variation: ",
#     round(metrics.explained_variance_score(Y_val, Y_val_pred), 2),
# )
# print("Validation MAPE:", round(get_mape(Y_val, Y_val_pred), 2))
# print(
#     "Validation Mean Squared Error:",
#     round(metrics.mean_squared_error(Y_train, Y_train_pred), 2),
# )
# print(
#     "Validation RMSE: ",
#     round(np.sqrt(metrics.mean_squared_error(Y_val, Y_val_pred)), 2),
# )
# print("Validation MAE: ", round(metrics.mean_absolute_error(Y_val, Y_val_pred), 2))

# print(" ")

# print("Test R-squared: ", round(metrics.r2_score(Y_test, Y_test_pred), 2))
# print(
#     "Test Explained Variation: ",
#     round(metrics.explained_variance_score(Y_test, Y_test_pred), 2),
# )
# print("Test MAPE:", round(get_mape(Y_test, Y_test_pred), 2))
# print(
#     "Test Mean Squared Error:",
#     round(metrics.mean_squared_error(Y_test, Y_test_pred), 2),
# )
# print("Test RMSE: ", round(np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred)), 2))
# print("Test MAE: ", round(metrics.mean_absolute_error(Y_test, Y_test_pred), 2))


# df_pred = pd.DataFrame(Y_val.values, columns=["Actual"], index=Y_val.index)
# df_pred["Predicted"] = Y_val_pred
# df_pred = df_pred.reset_index()
# df_pred.loc[:, "Date"] = pd.to_datetime(df_pred["Date"], format="%Y-%m-%d")
# df_pred


# df_pred[["Actual", "Predicted"]].plot()
