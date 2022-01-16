#!/usr/bin/env python3


import pandas as pd
import numpy as np
from pytrends import dailydata
import yfinance as yf
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

def main():
    print('Encoding google trends for %s ...' % google_trend)

    # LOAD PRICE DATA
    df_stock = yf.download('SPY', start='2016-01-01', end='3000-12-31', progress=False).reset_index(drop=False)[['Date', 'Open', 'Close']]
    df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')

    # LOAD GOOGLE TRENDS DATA
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

    df['date'] = df['date'].apply(lambda x: x.replace(tzinfo=None))
    df['date'] = pd.to_datetime(df['date'])


    # SHIFT TIME 

    # In prod, the last 14 hours can't be used yet (unstable). 5pm previous day must become 11pm of current day
    # (14 hours between 5pm previous and 7am current day)
    # we must add 30 hours to datetime to make 11pm of previous day the midnight of next day
    # but it doesn't mean we drop the last 30 hours of live GT! with time zones, it means we drop the last 14 hours only.
    df['date'] = df['date'] + pd.DateOffset(hours=30)


    # DEALING WITH MISSING VALUES

    # Replace missing values and zeros values by nans
    for col in df.columns[1:]:
        df.loc[(df[col].isna()) | (df[col] == 0), col] = np.nan


    # Replace 0s by -0.1 (so that whole days of zeros are negative but punctual zeros don't have any impact)
    df.loc[df[df.columns[1:].tolist()].sum(axis=1) == 0, df.columns[1:].tolist()] = -0.1


    # REMOVE WEEKENDS AND PUBLIC HOLIDAY
    df['Date'] = pd.to_datetime(df['date'].dt.date)
    df['time'] = df['date'].dt.time

    df = df.sort_values('date')

    df = df.merge(df_stock[['Date', 'Open']], on='Date', how='left')

    # create shift stock variables
    shift_cols = []
    for k in range(1, 20):
        df['stock_shift_%s' % k] = df.groupby('time')['Open'].shift(k)
        shift_cols.append('stock_shift_%s' % k)


    shift_var_cols = []
    for k in range(1, 20):
        df['var_shift_%s' % k] = df.groupby('time')[google_trend].shift(k)
        shift_var_cols.append('var_shift_%s' % k)

    df[google_trend] = df.apply(lambda row: correct_post_off_days_data(row, col, df['Date'].min()), axis=1)
    df = df.drop(shift_var_cols + shift_cols, axis=1)

    # Drop weekends / public hoidays (when no opening of stock market)
    df = df[(df['Open'].notna()) | (df['Date'] == df['Date'].max())]
    df = df.drop(['Open', 'time', 'Date'], axis=1)



    # DAILY AGGREGATION 
    df_d = df.copy()
    df_d['date'] = pd.to_datetime(df_d['date'].dt.date)
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

        # compute delta between GT of opening (10 am) and GT of closing (4pm)
        df_h_peaks = df.copy()
        df_h_peaks['hour'] = df_h_peaks['date'].dt.hour
        df_h_peaks = df_h_peaks[df_h_peaks['hour'].isin([16, 21])]   # matches for google trends of 10am, 4pm of previous day
        df_h_peaks['date'] = df_h_peaks['date'].dt.date
        df_h_peaks['hour'] = df_h_peaks['hour'].astype(str)
        df_h_peaks = pd.pivot_table(df_h_peaks, values=google_trend, index='date', columns='hour')
        df_h_peaks.columns = list(map("".join, df_h_peaks.columns))
        df_h_peaks['delta_GT_peaks'] = (1 - (df_h_peaks['16'] / df_h_peaks['21'])) * 100
        df_h_peaks['delta_GT_peaks'] = df_h_peaks['delta_GT_peaks'].fillna(-999)
        df_h_peaks = df_h_peaks.drop(['16', '21'], axis=1)
        df_h_peaks = df_h_peaks.reset_index(drop=False)

        # The GT on Monday must be the one obtained for previous saturday because GT on weekends are irrelevant
        def correct_monday_value_hourly(row):
            if row['day_of_week'] == 'Monday':
                return row['shift saturday']
            else:
                return row['delta_GT_peaks']

        df_h_peaks['date'] = pd.to_datetime(df_h_peaks['date'])
        df_h_peaks['day_of_week'] = df_h_peaks['date'].dt.day_name()
        df_h_peaks['shift saturday'] = df_h_peaks['delta_GT_peaks'].shift(2)
        df_h_peaks['delta_GT_peaks'] = df_h_peaks.apply(lambda row: correct_monday_value_hourly(row), axis=1)
        df_h_peaks = df_h_peaks.drop(['shift saturday', 'day_of_week'], axis=1)

        # remove holidays and weekends
        df_h_peaks = df_h_peaks[df_h_peaks.date.dt.dayofweek < 5]

        final_date = datetime.today()
        holidays = calendar().holidays(start='2000-01-01', end='%s-%s-%s' %(final_date.year, final_date.month, final_date.day))
        m = df_h_peaks['date'].isin(holidays)
        df_h_peaks = df_h_peaks[~m].copy()

        # create deltas
        df_h_peaks['delta_GT_peaks_lag1'] = df_h_peaks['delta_GT_peaks'].shift(1) / (1 + df_h_peaks['delta_GT_peaks'])
        df_h_peaks['delta_GT_peaks_lag2'] = df_h_peaks['delta_GT_peaks'].shift(2) / (1 + df_h_peaks['delta_GT_peaks'])
        df_h_peaks['delta_GT_peaks_lag3'] = df_h_peaks['delta_GT_peaks'].shift(3) / (1 + df_h_peaks['delta_GT_peaks'])

        # join hourly features on daily features 
        df = df_h_peaks.merge(df_d, on='date', how='inner')


    # if mood google trend, only daily aggregation
    else:
        df = df_d


    # Drop first nans due to shifts:
    df = df.rename(columns={'date': 'Date'})

    # save encoded features
    if os.path.isdir('./data/GOOGLE_TRENDS/%s/encoded_data' % google_trend) is False:
        os.system("mkdir './data/GOOGLE_TRENDS/%s/encoded_data'" % google_trend)

    df.to_csv('./data/GOOGLE_TRENDS/%s/encoded_data/%s_features_google.csv' % (google_trend, google_trend), index=False)


def correct_post_off_days_data(row, col, min_date):

    # first days have nans stock values (will be dropped in the end)
    if row['Date'] <= min_date + pd.DateOffset(days=10):
        return row[col]

    if np.isnan(row['stock_shift_1']) == False:
        return row[col]

    else:
        max_depth = 10
        depth = 1
        while np.isnan(row['stock_shift_%s' % depth]) == True:
            depth = depth + 1

            if depth >= max_depth:
                raise ValueError('More than %s days in a row had NaN stock prices' % max_depth) 

        return row['var_shift_%s' % (depth - 1)] #  depth - 1 because we want last row before it's no longer nan


if __name__ == "__main__":
    main()

