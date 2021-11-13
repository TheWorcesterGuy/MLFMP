import pandas as pd
import numpy as np
from pytrends import dailydata
from datetime import datetime
from pytrends.request import TrendReq
import pytrends
from pathlib import Path
import time
import random
import urllib.request
import matplotlib.pyplot as plt
import os
import argparse
import glob
import pytz
import csv

google_trends = ['facebook stock', 'SPY', 'AMD', 'AAPL', 'AMZN', 'QQQ', 'TSLA', 'MSFT', 'boeing stock',
                  'INTC', 'DIS', 'JPM', 'WMT', 'NFLX', 'GOOG', 'GOOGL', 'NVDA', 'TWTR',
                  'debt', 'bloomberg', 'yahoo finance', 'buy stocks', 'sell stocks', 'VIX', 'stock risk']


for google_trend in google_trends:

    files = glob.glob("./data/GOOGLE_TRENDS/%s/*.csv" % google_trend)

    # get the current Google Trend file so that we don't fill the holes in the currently updating one
    current_year = 0
    current_month = 0
    for file in files:
        if int(file.split('_')[3]) > current_year:
            current_year = int(file.split('_')[3])
    for file in files:
        if int(file.split('_')[3]) == current_year:
            if int(file.split('_')[4][:-4]) > current_month:
                current_month = int(file.split('_')[4][:-4])
    current_file = './data/GOOGLE_TRENDS/%s/Google_Trends_%s_%s.csv' % (google_trend, current_year, current_month)

    for file in files:

        # we don't want to fill holes on the current month but on historical files
        if file == current_file:
            pass

        else:
            df_file = pd.read_csv(file)

            percentage_zeros = np.round(df_file[df_file[google_trend] == 0].shape[0] / df_file.shape[0] * 100, 1)

            if percentage_zeros > 10:
                print('%s:' % file)
                print('%s%% of missing values' % percentage_zeros)

                # convert datetime back to UTC (google trends is based on UTC)
                df_file['date'] = pd.to_datetime(df_file['date'])

                date_start = pd.to_datetime(df_file['date'].min())
                date_end = pd.to_datetime(df_file['date'].max())

                date_start = date_start.tz_convert('UTC')
                date_end = date_end.tz_convert('UTC')

                os.system(
                    "python query_google.py %s %s %s %s %s %s %s" % (date_start.year, date_start.month, date_start.day,
                                                                     date_end.year, date_end.month, date_end.day,
                                                                     str(google_trend)))
                df_new_file = pd.read_csv('data/temp_data.csv')

                new_percentage_zeros = np.round(df_new_file[df_new_file[google_trend] == 0].shape[0] / df_new_file.shape[0] * 100, 1)

                if new_percentage_zeros < percentage_zeros:
                    df_new_file.to_csv(file, index=False)
                    os.system('rm data/temp_data.csv')
                    print('New file saved with %s%% less missing values.\n' % (percentage_zeros - new_percentage_zeros))

                else:
                    print("The new file was similar.\n")

                time.sleep(15)


