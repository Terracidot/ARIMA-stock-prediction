import pandas as pd
import datetime as datetime
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.dates as mpldates
import pandas_datareader.data as web
import seaborn as sns

#print(df.head())
#print(df.describe())
#print(df.columns)
#print(df.info())
#print(df['ID'].value_counts())
#print(df.shape)

# Variables
company_name = 'atvi'
resource = 'yahoo'
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.now()

# Graph style
style.use('ggplot')

# Read in data
df = web.DataReader(company_name, resource, start, end)
df1 = pd.DataFrame(data=df)
# print(df1)

# Moving average
df1['100_move_avg'] = df1['Adj Close'].rolling(window=100, min_periods=0).mean()
# print(df1)

df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_vol = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mpldates.date2num)
print(df_ohlc)

# plotting
ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=5, colspan=1, sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
ax2.fill_between(df_vol.index.map(mpldates.date2num), df_vol.values, 0)

# df2 = df1.corr()
# sns.heatmap(data=df2, cmap='magma', linecolor='white', linewidths=0.5, annot=False)

plt.tight_layout()
plt.show()

