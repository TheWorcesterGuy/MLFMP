import pandas as pd
import numpy as np
from pytrends import dailydata
from datetime import datetime, timedelta
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
from pathlib import Path
from pytz import timezone


def main():

    waiting_time = 30
    n_days = 14

    # Check if Google Trend update is allowed
    check_update_validity()

    google_trends = ['INTC', 'TSLA', 'AMZN', 'FB', 'AAPL', 'DIS', 'SPY', 'QQQ', 'GOOG', 'GOOGL', 'MSFT', 'NFLX', 'NVDA',
        	  'TWTR', 'AMD', 'WMT', 'JPM', 'BAC', 'PG', 'VIX', 'debt', 'bloomberg', 'yahoo finance', 'buy stocks', 'sell stocks', 'stock risk',
                     'investing.com', ['bullish', 'bearish']]

    if os.path.isfile('./data/temp_data.csv'):
        os.system('rm ./data/temp_data.csv')

    default_dir = os.getcwd()

    for google_trend in google_trends:

        # for dual Bullish / Bearish
        if type(google_trend) == list:

            google_trend_str = '_'.join(google_trend)
            print('Downloading Google Trends for %s...' % google_trend_str)

            if os.path.isdir('./data/GOOGLE_TRENDS/%s' % google_trend_str) is False:
                os.system("mkdir './data/GOOGLE_TRENDS/%s'" % google_trend_str)

            files = glob.glob("./data/GOOGLE_TRENDS/%s/*.csv" % google_trend_str)
            files = [file for file in files if 'encoded' not in file]

            if len(files) > 0:

                # first, remove the last 100 GT as we want them to be replaced by the older ones (recent GT are still unstable)
                list_updt = [pd.read_csv(file) for file in glob.glob("./data/GOOGLE_TRENDS/%s/*.csv" % google_trend_str)]
                df_updt = pd.concat(list_updt, axis=0)
                df_updt['date'] = pd.to_datetime(df_updt['date'], errors='coerce')
                df_updt = df_updt.sort_values('date', ascending=True)
                df_updt = df_updt.iloc[0:-100]
                repartition_files(df_updt, google_trend_str, default_dir)

                # find the latest datetime and resume downloading
                list_gt = [pd.read_csv(file, lineterminator='\n') for file in
                           glob.glob("./data/GOOGLE_TRENDS/%s/*.csv" % google_trend_str)]
                df = pd.concat(list_gt, axis=0)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                start_date = df['date'].max()
            else:
                eastern = pytz.timezone('US/Eastern')
                start_date = pd.to_datetime('2016-01-01', format='%Y-%m-%d').tz_localize(pytz.utc).tz_convert(eastern)

            final_date = datetime.now(pytz.timezone('US/Eastern'))
            nb_try = 0
            last_file_saved = False

            while last_file_saved is False:

                if start_date + pd.DateOffset(days=n_days) > final_date:
                    next_date = final_date + pd.DateOffset(days=1)
                    start_date = final_date - pd.DateOffset(days=n_days)
                    filename = './data/GOOGLE_TRENDS/bullish_bearish/current_file.csv'
                    last_file = True

                else:
                    next_date = start_date + pd.DateOffset(days=n_days)
                    filename = 'data/GOOGLE_TRENDS/bullish_bearish/Google_Trends_%s_week_%s.csv' % (next_date.year, next_date.week)
                    last_file = False

                while os.path.isfile('data/temp_data.csv') is False:

                    os.system("python query_google.py %s %s %s %s %s %s %s %s" % (start_date.year, start_date.month, start_date.day,
                                                                                   next_date.year, next_date.month, next_date.day, next_date.hour, str(google_trend)))

                df = pd.read_csv('data/temp_data.csv')
                df['isPartial'] = df['isPartial'].astype(str)

                last_file_saved = datetime.now(pytz.timezone('US/Eastern')) - timedelta(hours=1) <= pd.to_datetime(df['date'].max())

                if last_file is True and last_file_saved is False:
                    nb_try = nb_try + 1

                if nb_try > 3:
                    last_file_saved = 1
                    print('\nFailed to fully save data for current month!\n')

                df = df[df['isPartial'] == 'False']
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] < datetime.now(pytz.timezone('US/Eastern')) - pd.Timedelta(hours=1)]

                print('Last row downloaded : ', df) #.tail(1))

                df.to_csv(filename, index=False)
                start_date = df['date'].max()

                os.system('rm data/temp_data.csv')
                time.sleep(waiting_time)

            # repartition by month the updated files
            list_updt = [pd.read_csv(file) for file in glob.glob("./data/GOOGLE_TRENDS/%s/*.csv" % google_trend_str)]
            df_updt = pd.concat(list_updt, axis=0)
            repartition_files(df_updt, google_trend_str, default_dir)
            
            print('Done downloading Google Trends for %s.\n\n' % google_trend)
            print('\n\n')

        # For single keywords
        else:
        
            print('Downloading Google Trends for %s...' % google_trend)

            if os.path.isdir('./data/GOOGLE_TRENDS/%s' % google_trend) is False:
                os.system("mkdir './data/GOOGLE_TRENDS/%s'" % google_trend)

            files = glob.glob("./data/GOOGLE_TRENDS/%s/*.csv" % google_trend)
            files = [file for file in files if 'encoded' not in file]

            if len(files) > 0:

                # first, remove the last 100 GT as we want them to be replaced by the older ones (recent GT are still unstable)
                list_updt = [pd.read_csv(file) for file in glob.glob("./data/GOOGLE_TRENDS/%s/*.csv" % google_trend)]
                df_updt = pd.concat(list_updt, axis=0)
                df_updt['date'] = pd.to_datetime(df_updt['date'], errors='coerce')
                df_updt = df_updt.sort_values('date', ascending=True)
                df_updt = df_updt.iloc[0:-100]
                repartition_files(df_updt, google_trend, default_dir)

                # find the latest datetime and resume downloading
                list_gt = [pd.read_csv(file, lineterminator='\n') for file in
                           glob.glob("./data/GOOGLE_TRENDS/%s/*.csv" % google_trend)]
                df = pd.concat(list_gt, axis=0)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                start_date = df['date'].max()
            
            else:
                eastern = pytz.timezone('US/Eastern')
                start_date = pd.to_datetime('2016-01-01', format='%Y-%m-%d').tz_localize(pytz.utc).tz_convert(eastern)
    

            final_date = datetime.now(pytz.timezone('US/Eastern'))
            nb_try = 0
            last_file_saved = False

            while last_file_saved is False:

                if start_date + pd.DateOffset(days=n_days) > final_date:
                    next_date = final_date + pd.DateOffset(days=1)
                    start_date = final_date - pd.DateOffset(days=n_days)
                    filename = './data/GOOGLE_TRENDS/%s/current_file.csv' % google_trend
                    last_file = True

                else:
                    next_date = start_date + pd.DateOffset(days=n_days)
                    filename = 'data/GOOGLE_TRENDS/%s/Google_Trends_%s_week_%s.csv' % (google_trend, next_date.year, next_date.week)
                    last_file = False

                while os.path.isfile('data/temp_data.csv') is False:
                    os.system("python query_google.py %s %s %s %s %s %s %s %s" % (start_date.year, start_date.month, start_date.day,
                                                                                   next_date.year, next_date.month, next_date.day, next_date.hour, str(google_trend)))
                df = pd.read_csv('data/temp_data.csv')
                df['isPartial'] = df['isPartial'].astype(str)
                
                # check if last download                
                last_file_saved = datetime.now(pytz.timezone('US/Eastern')) - timedelta(hours=1) <= pd.to_datetime(df['date'].max())

                if last_file is True and last_file_saved is False:
                    nb_try = nb_try + 1

                if nb_try > 3:
                    last_file_saved = 1
                    print('\nFailed to fully save data for current month!\n')

                df = df[df['isPartial'] == 'False']
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] < datetime.now(pytz.timezone('US/Eastern')) - pd.Timedelta(hours=1)]

                print('Last row downloaded : ', df) #.tail(1))
                
                df.to_csv(filename, index=False)
                start_date = df['date'].max()

                os.system('rm data/temp_data.csv')
                time.sleep(waiting_time)

            # repartition by month the updated files
            list_updt = [pd.read_csv(file) for file in glob.glob("./data/GOOGLE_TRENDS/%s/*.csv" % google_trend)]
            df_updt = pd.concat(list_updt, axis=0)
            repartition_files(df_updt, google_trend, default_dir)

            print('Done downloading Google Trends for %s.\n\n' % google_trend)
            print('\n\n')

    with open('data/history_google_updates.csv', 'a') as fd:
        fd.write(pd.Timestamp.now(tz='US/Eastern').strftime('%Y-%m-%d') + '\n')


def check_update_validity():
    # load historical google updates
    file = open("data/history_google_updates.csv", "r")
    csv_reader = csv.reader(file)
    update_dates = []
    for row in csv_reader:
        update_dates.append(row[0])

    # get current date and check if a request has already been made today
    current_date = pd.Timestamp.now(tz='US/Eastern').strftime('%Y-%m-%d')

    if current_date in update_dates:
        print('A Google Trend update has already been done today between.')
        exit(0)

    # if during_week_days
    if pd.to_datetime(datetime.now().date()).dayofweek < 5:
        # get current date
        hour = pd.Timestamp.now(tz='US/Eastern').hour
        minute = pd.Timestamp.now(tz='US/Eastern').minute

        if (hour == 8 and 10 <= minute <= 50) is False:
            print('Downloading of Google Trends forbidden : must occur between 8:10am and 8:50am NY time only.')
            exit(0)
        else:
            pass  # update request accepted

    else:
        print("This is the weekend : running 'fill_google_trend_holes.py'")
        exit(0)
        #os.system('python fill_google_trend_holes.py')

def repartition_files(df_updt, google_trend, default_dir):

    df_updt['fixed_date'] = df_updt['date'].astype(str)
    df_updt['fixed_date'] = df_updt['fixed_date'].str[:10]
    df_updt['fixed_date'] = pd.to_datetime(df_updt['fixed_date'], errors='coerce')
    df_updt['year'] = df_updt['fixed_date'].dt.year
    df_updt['month'] = df_updt['fixed_date'].dt.month

    years = df_updt['year'].unique()
    months = df_updt['month'].unique()

    os.chdir('data/GOOGLE_TRENDS/%s/' % google_trend)
    os.system("rm *.csv")
    os.chdir(default_dir)

    for year in years:
        df_year = df_updt[df_updt['year'] == year]
        for month in months:
            df_year_month = df_year[df_year['month'] == month]
            if df_year_month.empty is False:
                # if two duplicates for same date, we keep the one with larger value (avoid adding zeros)
                df_year_month = df_year_month.sort_values(by=df_year_month.columns[:2].tolist(), ascending=True)
                df_year_month = df_year_month.drop_duplicates(subset=['date'], keep='last')
                # save file
                df_year_month = df_year_month[df_year_month.columns[:-3].tolist()]
                df_year_month.to_csv('data/GOOGLE_TRENDS/%s/Google_Trends_%s_%s.csv' % (google_trend, year, month),
                                                index=False)

if __name__ == "__main__":
    main()
