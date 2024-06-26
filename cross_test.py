import pandas as pd
import datetime
import math
from matplotlib import style
import numpy as np
from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.dates as mpldates
import pandas_datareader.data as web
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics, preprocessing
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn import cross_decomposition

# Undo scaling
# var_name = scaler.inverse_transform(pred_price)

# # Create df of dates equal to length + n days
# rng = pd.date_range(start, end, freq='B')
# date_df = pd.DataFrame({'Date': rng, 'Adj Close': np.random.randn(len(rng))})
# date_df.set_index('Date', inplace=True)
# print(date_df)

# Variables
company_name = 'atvi'
resource = 'yahoo'
start = datetime.datetime(1990, 1, 1)
end = datetime.datetime.now()
days_to_predict = 10
prediction_col = 'Adj Close'

# Read in data
df1 = web.DataReader(company_name, resource, start, end)
df1 = pd.DataFrame(data=df1)
# print(df1)

# Add moving average. window equals number of days
df1['moving_avg'] = df1['Adj Close'].rolling(window=100, min_periods=0).mean()
# Add high/low percentage change
df1['high_low_percent'] = (df1['High'] - df1['Adj Close']) / df1['Adj Close'] * 100.0
# Add daily percentage change
df1['daily_percent'] = (df1['Adj Close'] - df1['Open']) / df1['Open'] * 100.0
# Select needed columns
df1 = df1[['Adj Close', 'high_low_percent', 'daily_percent', 'Volume']]
# Fill nan values
df1.fillna(-99999, inplace=True)

# Select number of days to shift prediction column up

# print(prediction_out)

# Shift columns negatively according to number of days for prediction
df1['Prediction'] = df1[prediction_col].shift(-days_to_predict)

# # Define x and y features and labels
# x features
x = np.array(df1.drop(['Prediction'], axis=1))
# Scaling data. Note! Do not use scaling for high frequency trading
x = preprocessing.scale(x)
x_lately = x[-days_to_predict:]

print('x_lately', x_lately)
x = x[:-days_to_predict]
print(len(x))

print(len(x_lately))

df1.dropna(inplace=True)
y = np.array(df1['Prediction'])
print('sssssssssssssssssss', y)
# Split for train, test, split
print('df1', df1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Linear regression model
lr_model = LinearRegression().fit(x_train, y_train)
lr_score = lr_model.score(x_test, y_test)
print("LR estimated accuracy of", lr_score * 100, "%")

# Support vector model
svm_model = SVR().fit(x_train, y_train)
svr_score = svm_model.score(x_test, y_test)
print("SVM estimated accuracy of", svr_score * 100, "%")

forecast_set = lr_model.predict(x_lately)
print('forecast_set', forecast_set)

# Graph style
style.use('ggplot')

# Fill with nan values
df1['Forecast'] = np.nan
# Get last date of timestamp
last_date = df1.iloc[-1].name
last_unix = last_date.timestamp()
# Number of seconds in one day
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df1.loc[next_date] = [np.nan for _ in range(len(df1.columns)-1)] + [i]

df1['Adj Close'].plot()
df1['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')

# plt.tight_layout()
# plt.show()
