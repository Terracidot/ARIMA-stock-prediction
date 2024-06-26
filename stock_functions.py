import pandas as pd
from datetime import timedelta, datetime
import math
from matplotlib import style
import numpy as np
from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.dates as mpldates
import pandas_datareader.data as web
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import cross_decomposition
from sklearn import preprocessing
import pmdarima as pm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
import itertools
import warnings
warnings.filterwarnings('ignore')

days_to_predict = 0
company_name = 'atvi'
resource = 'yahoo'
start = datetime(2000, 1, 1)
end = datetime.now()
prediction_col = 'Adj Close'

# Read in data
df1 = web.DataReader(company_name, resource, start, end)
df1 = pd.DataFrame(data=df1)
# print(df1)

# Create df of n days
rng = pd.date_range(end + timedelta(days=1), end + timedelta(days=days_to_predict), freq='B')
date_df = pd.DataFrame({'Date': rng})  # df1['Adj Close']})
date_df.set_index('Date', inplace=True)
# print(date_df)

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

# df_hlp = df1['high_low_percent']
# print(df_hlp)

# Data scaling
df_hlp = pd.DataFrame(df1['high_low_percent'])
# print(df_hlp)
scaler = StandardScaler()
df_hlp[['high_low_percent']] = scaler.fit_transform(df_hlp[['high_low_percent']])
df_hlp = pd.Series(df_hlp['high_low_percent'])

df_con = pd.concat([df_hlp, df_hlp.shift(3)], axis=1)
df_con.columns = ['high_low_percent', 'forecast percent']
df_con.dropna(inplace=True)


# print(df_con)

# ARIMA functions

def plot_acf_pacf(series):
    # q = 4, p = 3, d = 0-2
    plot_acf(series[-101:])  # Determine q value
    plot_pacf(series[-101:])  # Determine p value

    plt.tight_layout()
    plt.show()

def find_best_pdq(train_data):
    p = d = q = range(0, 5)
    pdq = list(itertools.product(p, q, d))

    for parameters in pdq:
        try:
            arima_model = ARIMA(train_data, order=parameters).fit()
            print('*******', parameters, arima_model.aic, '*******')
        except:
            continue




