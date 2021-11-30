#!/usr/bin/env python3


import pandas as pd
import numpy as np
from pytrends import dailydata
from datetime import datetime
import pytrends
from pathlib import Path
import time
import random
import glob
from functools import reduce
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import os
from functools import reduce
import sys


google_trend = sys.argv[1]


print('Encoding google trends for %s ...' % google_trend)

files = glob.glob("./data/GOOGLE_TRENDS/%s/*.csv" % google_trend)
files = [file for file in files if 'encoded' not in file]

if len(files) > 0:
    list_gt = [pd.read_csv(file, lineterminator='\n') for file in files]
    df = pd.concat(list_gt, axis=0)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.drop_duplicates('date')
    del df['isPartial']

else:
    print('Error : empty folder for %s' % google_trend)
    exit(0)

# Google trends after 8:00 am (NY time) included are considered for the next day, which means we shift forward by 16 hours
df['date'] = df['date'].apply(lambda x: x.replace(tzinfo=None))
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'] + pd.DateOffset(hours=16)

# remove holidays and weekends
df = df[df.date.dt.dayofweek < 5]

final_date = datetime.today()
holidays = calendar().holidays(start='2000-01-01', end='%s-%s-%s' %(final_date.year, final_date.month, final_date.day))
m = df['date'].isin(holidays)
df = df[~m].copy()


# Create a time series with all timestamp expect weekends & holidays (used to detected missing records in Google trends)
time = pd.DataFrame(pd.date_range(df['date'].min(), df['date'].max(), freq='H').values,
                    columns=['date'])
m = time['date'].isin(holidays)
time = time[~m].copy().sort_values('date', ascending=False)
time = time[time.date.dt.dayofweek < 5]

df = df.merge(time['date'], on='date', how='outer').sort_values('date', ascending=False)


# Replace missing values and zeros values by nans
for col in df.columns[1:]:
    df.loc[(df[col].isna()) | (df[col] == 0), col] = np.nan


# Replace 0s by -0.1 (so that whole days of zeros are negative but punctual zeros don't have any impact)
df.loc[df[google_trend] == 0, google_trend] = -0.1

# Average Google trends daily
df['date'] = df['date'].dt.date
df = df.groupby('date', as_index=False).mean()


# Rename stock names with generic name 'stock'
for col in df.columns:
    if col in ['facebook stock', 'SPY', 'AMD', 'AAPL', 'AMZN', 'QQQ', 'TSLA', 'MSFT', 'boeing stock',
               'INTC', 'DIS', 'JPM', 'WMT', 'NFLX', 'GOOG', 'GOOGL', 'NVDA', 'TWTR']:
        df.rename(columns={col: 'stock'}, inplace=True)

# Create delta features
for col in df.columns[1:]:

    df['GT_%s_delta1' % col] = (df[col] - df[col].shift(1)) / (df[col].shift(1) + 1)
    df['GT_%s_delta2' % col] = (df[col] - df[col].shift(2)) / (df[col].shift(2) + 1)
    df['GT_%s_delta3' % col] = (df[col] - df[col].shift(3)) / (df[col].shift(3) + 1)
    df['GT_%s_delta4' % col] = (df[col] - df[col].shift(4)) / (df[col].shift(4) + 1)
    df['GT_%s_delta1_smooth' % col] = (df[col] + df[col].shift(1)) / (df[col].shift(2) + df[col].shift(3) + 1)
    df['GT_%s_delta2_smooth' % col] = (df[col].shift(2) + df[col].shift(3)) / (df[col].shift(4) + df[col].shift(5) + 1)
    df['GT_%s' % col] = df[col]
    del df[col]
    

# Drop first nans due to shifts:
df = df.rename(columns={'date': 'Date'})


# save encoded features
if os.path.isdir('./data/GOOGLE_TRENDS/%s/encoded_data' % google_trend) is False:
    os.system("mkdir './data/GOOGLE_TRENDS/%s/encoded_data'" % google_trend)

df.to_csv('./data/GOOGLE_TRENDS/%s/encoded_data/%s_features_google.csv' % (google_trend, google_trend), index=False)


