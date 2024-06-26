import pandas as pd
import datetime as datetime
import math as math
from matplotlib import style
import numpy as np
from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.dates as mpldates
import pandas_datareader.data as web
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV

# Variables
company_name = 'atvi'
resource = 'yahoo'
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2020, 6, 30)
days = 5

# Undo scaling
# var_name = scaler.inverse_transform(pred_price)

# Graph style
style.use('ggplot')

# Read in data
df = web.DataReader(company_name, resource, start, end)
df1 = pd.DataFrame(data=df)
print(df1)
length = len(df1)+days

# Create df of dates equal to length + n days
rng = pd.date_range(start, end, freq='B')
date_df = pd.DataFrame({'Date': rng, 'Adj Close': np.random.randn(len(rng))})
date_df.set_index('Date', inplace=True)
print(date_df)

# 110 day moving average
df1['100_move_avg'] = df1['Adj Close'].rolling(window=100, min_periods=0).mean()

df1['Prediction'] = df1['Adj Close'].shift(-days)
# print(df1)
# print(df1.columns)

df1 = df1[['Prediction', 'Adj Close']]
print(df1)

x = np.array(df1.drop('Prediction', axis=1))
x = x[:-days]
# print(x)

y = np.array(df1['Prediction'])
# Get all values except n days
y = y[:-days]
# print(y)

# Split for train, test, split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Pass values of 'C' and 'gamma' through grid search for hyper-parameter optimality of SVM
# grid_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
#                       'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
#                       'kernel': ['rbf', 'linear', 'poly']}
#
# grid = GridSearchCV(SVR(kernel='rbf'), grid_parameters, verbose=1000).fit(x_train, y_train)
#
# print("Best hyper parameters", grid.best_params_)
# print("Best estimator", grid.best_estimator_)
#
# grid_prediction = grid.predict(x_)
# print(grid_prediction)

# Support vector model
svm_model = SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=1e-05,
                kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False).fit(x_train, y_train)
svr_score = svm_model.score(x_test, y_test)
print("SVM estimated accuracy of", svr_score * 100, "%")

# Linear regression model
lr_model = LinearRegression().fit(x_train, y_train)
lr_score = lr_model.score(x_test, y_test)
print("LR estimated accuracy of", lr_score * 100, "%")

# Set x_forecast equal to last n rows of adj close
x_forecast = np.array(df1.drop('Prediction', axis=1))[-days:]
# print(x_forecast)

# SVM predictions
svm_prediction = svm_model.predict(x_forecast)
print(days, "day SVM prediction", svm_prediction)

# LR predictions
lr_prediction = lr_model.predict(x_forecast)
print(days, "day LR prediction", lr_prediction)






