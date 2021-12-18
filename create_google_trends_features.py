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
df.loc[df[df.columns[1:].tolist()].sum(axis=1) == 0, df.columns[1:].tolist()] = -0.1




# DAILY AGGREGATION (AVERAGE)
df_d = df.copy()
df_d['date'] = df_d['date'].dt.date
df_d = df_d.groupby('date', as_index=False).mean()


# Rename stock names with generic name 'stock'
for col in df_d.columns:
    if col in ['INTC', 'TSLA',  'AMZN', 'FB', 'AAPL', 'DIS', 'SPY', 'QQQ', 'GOOG', 'GOOGL', 'MSFT', 'NFLX', 'NVDA',
              'TWTR', 'AMD', 'WMT', 'JPM', 'BAC', 'PG']:
        df_d.rename(columns={col: 'stock'}, inplace=True)

# Create delta features
# for couple keywords
if google_trend == 'bullish_bearish':
    df_d['delta_bullish_bearish'] = df_d['bullish'] - df_d['bearish']
    for col in df_d.columns[1:]:

        df_d['GT_%s_delta1' % col] = (df_d[col] - df_d[col].shift(1)) / (df_d[col].shift(1) + 1)
        df_d['GT_%s_delta2' % col] = (df_d[col] - df_d[col].shift(2)) / (df_d[col].shift(2) + 1)
        df_d['GT_%s_delta3' % col] = (df_d[col] - df_d[col].shift(3)) / (df_d[col].shift(3) + 1)
        df_d['GT_%s_delta4' % col] = (df_d[col] - df_d[col].shift(4)) / (df_d[col].shift(4) + 1)
        df_d['GT_%s_delta1_smooth' % col] = (df_d[col] + df_d[col].shift(1)) / (df_d[col].shift(2) + df_d[col].shift(3) + 1)
        df_d['GT_%s_delta2_smooth' % col] = (df_d[col].shift(2) + df_d[col].shift(3)) / (df_d[col].shift(4) + df_d[col].shift(5) + 1)
        df_d['GT_%s' % col] = df_d[col]


# for single keywords:
else:
    for col in df_d.columns[1:]:
        df_d['GT_%s_delta1' % col] = (df_d[col] - df_d[col].shift(1)) / (df_d[col].shift(1) + 1)
        df_d['GT_%s_delta2' % col] = (df_d[col] - df_d[col].shift(2)) / (df_d[col].shift(2) + 1)
        df_d['GT_%s_delta3' % col] = (df_d[col] - df_d[col].shift(3)) / (df_d[col].shift(3) + 1)
        df_d['GT_%s_delta4' % col] = (df_d[col] - df_d[col].shift(4)) / (df_d[col].shift(4) + 1)
        df_d['GT_%s_delta1_smooth' % col] = (df_d[col] + df_d[col].shift(1)) / (df_d[col].shift(2) + df_d[col].shift(3) + 1)
        df_d['GT_%s_delta2_smooth' % col] = (df_d[col].shift(2) + df_d[col].shift(3)) / (
                    df_d[col].shift(4) + df_d[col].shift(5) + 1)
        df_d['GT_%s' % col] = df_d[col]
        del df_d[col]




# HOURLY AGGREGATION (only for stock keywords)

if google_trend in ['INTC', 'TSLA',  'AMZN', 'FB', 'AAPL', 'DIS', 'SPY', 'QQQ', 'GOOG', 'GOOGL', 'MSFT', 'NFLX', 'NVDA',
              'TWTR', 'AMD', 'WMT', 'JPM', 'BAC', 'PG']:

	# compute delta between GT of opening (9 am) and GT of closing (3pm)
	df_h_peaks = df.copy()
	df_h_peaks['hour'] = df_h_peaks['date'].dt.hour
	df_h_peaks = df_h_peaks[df_h_peaks['hour'].isin([1, 7])]   # matches for google trends of 9am, 3pm of previous day
	df_h_peaks['date'] = df_h_peaks['date'].dt.date
	df_h_peaks['hour'] = df_h_peaks['hour'].astype(str)
	df_h_peaks = pd.pivot_table(df_h_peaks, values=google_trend, index='date', columns='hour')
	df_h_peaks.columns = list(map("".join, df_h_peaks.columns))
	df_h_peaks['delta_GT_peaks'] = (1 - (df_h_peaks['1'] / df_h_peaks['7'])) * 100
	df_h_peaks['delta_GT_peaks'] = df_h_peaks['delta_GT_peaks'].fillna(-999)
	df_h_peaks = df_h_peaks.drop(['1', '7'], axis=1)


	# Compute opening Google Trends (6am and 7am)
	df_h_opening = df.copy()
	df_h_opening['hour'] = df_h_opening['date'].dt.hour
	df_h_opening = df_h_opening[df_h_opening['hour'].isin([22, 23])]   # matches for google trends of 6am, 7pm of current
	df_h_opening['date'] = df_h_opening['date'].dt.date
	df_h_opening['hour'] = df_h_opening['hour'].astype(str)
	df_h_opening = pd.pivot_table(df_h_opening, values=google_trend, index='date', columns='hour')
	df_h_opening.columns = list(map("".join, df_h_opening.columns))
	df_h_opening['mean_GT_opening'] = (df_h_opening['22'] + df_h_opening['23']) / 2
	df_h_opening = df_h_opening.drop(['22', '23'], axis=1)

	# create deltas
	df_h_opening['mean_GT_opening_delta1'] = (df_h_opening['mean_GT_opening'] - df_h_opening['mean_GT_opening'].shift(1)) / (df_h_opening['mean_GT_opening'] + 1)
	df_h_opening['mean_GT_opening_delta2'] = (df_h_opening['mean_GT_opening'] - df_h_opening['mean_GT_opening'].shift(2)) / (df_h_opening['mean_GT_opening'] + 1)
	df_h_opening = df_h_opening.drop('mean_GT_opening', axis=1)

	# merge both features hourly together
	df_h = df_h_peaks.merge(df_h_opening, on='date', how='inner')
	
	
	# join hourly features on daily features 
	df = df_h.merge(df_d, on='date', how='inner')



# Drop first nans due to shifts:
df = df.rename(columns={'date': 'Date'})


# save encoded features
if os.path.isdir('./data/GOOGLE_TRENDS/%s/encoded_data' % google_trend) is False:
    os.system("mkdir './data/GOOGLE_TRENDS/%s/encoded_data'" % google_trend)

df.to_csv('./data/GOOGLE_TRENDS/%s/encoded_data/%s_features_google.csv' % (google_trend, google_trend), index=False)


