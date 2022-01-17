#!/usr/bin/env python3

import pandas as pd
import numpy as np
import glob
import os
import sys
import subprocess
import time
from datetime import datetime, timedelta
from functools import reduce

import warnings
warnings.simplefilter(action='ignore')


"""
DOCUMENTATION


The update of the feature store can be launch at any time but the program might get paused in order to respect the following constraints

-	Tweets download can occur only after minute 35. This ensures that the daily tweets are fully downloaded in everyday prod scenario. 
-	Google Trends must be downloaded after minute 10. This ensures that the previous hour Google Trend is fully ready for download.

The current recommended execution of this program is :

-	First update of the feature store at 7:40 to download most of the tweets.
-	Second update of the feature store at 8:10 to download the Google Trends and downloading the last tweets of the day.
"""


def main():

    update_start_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M')
    print('UPDATE START DATE (UTC) : %s\n\n' % update_start_date)

    stocks = ['INTC', 'TSLA',  'AMZN', 'FB', 'AAPL', 'DIS', 'SPY', 'QQQ', 'GOOG', 'GOOGL', 'MSFT', 'NFLX', 'NVDA',
              'TWTR', 'AMD', 'WMT', 'JPM', 'BAC', 'PG']

    google_trends_dir = ['INTC', 'TSLA', 'AMZN', 'FB', 'AAPL', 'DIS', 'SPY', 'QQQ', 'GOOG', 'GOOGL', 'MSFT', 'NFLX', 'NVDA',
        	  'TWTR', 'AMD', 'WMT', 'JPM', 'BAC', 'PG', 'debt', 'bloomberg', 'yahoo finance', 'buy stocks', 'sell stocks', 'VIX', 'stock risk',
                    'investing.com', 'bullish_bearish']

    start = datetime.now()
    t0 = time.time()
    
    # update minute price data download and create features
    os.system('python download_minute_price.py')
    apply_parallel_command(5, "./create_minute_price_features.py", stocks)
    
    stop_high_res_price = datetime.now()

    # update google trends download and create features
    wait_for_time(10, 'Google Trends')
    os.system("python download_google_trends.py")
    apply_parallel_command(10, "./create_google_trends_features.py", google_trends_dir)

    stop_google_trends = datetime.now()

    # update download price data and create features
    os.system("python download_price.py")
    apply_parallel_command(5, "./create_trading_features.py", stocks)

    stop_trade = datetime.now()

    # update tweet download and create features
    wait_for_time(35, 'Tweets')
    apply_parallel_command(6, "./download_tweets.py", stocks)
    apply_parallel_command(6, "./encode_tweets.py", stocks)
    apply_parallel_command(3, "./create_twitter_features.py", stocks)
    

    # merge the two sources of features
    df = merge_files(stocks)
    df = df.set_index(['Date', 'stock'])
    df.to_csv('./data/features_store.csv')
    print('features store updated')


    stop = datetime.now()
    
    print('\n Time to update minute price features: ', (stop_high_res_price - start))
    print('\n Time to update google trend features: ', (stop_google_trends - start))
    print('\n Time to update trade features: ', (stop_trade - stop_google_trends))
    print('\n Time to update twitter features: ', (stop - stop_trade))
    print('\n Time to update features: ', (stop - start))
    df = df.reset_index(drop=False)
    print('\nLast rows added:', df[df['Date'] == df['Date'].max()])
    
    
    update_end_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M')
    print('\n\nUPDATE END DATE (UTC) : %s\n\n' % update_end_date)
    
    os.system('python reporting_features_store.py')

    

def apply_parallel_command(max_processes, command, list_argument, env_path=None):

    processes = set()

    if env_path:

        for name in list_argument:
            processes.add(subprocess.Popen([command, name, env_path]))
            if len(processes) >= max_processes:
                os.wait()
                processes.difference_update(
                    [p for p in processes if p.poll() is not None])

    else:

        for name in list_argument:
            processes.add(subprocess.Popen([command, name]))
            if len(processes) >= max_processes:
                os.wait()
                processes.difference_update(
                    [p for p in processes if p.poll() is not None])

    #Check if all the child processes were closed
    for p in processes:
        if p.poll() is None:
            p.wait()


def merge_files(stocks):
    df_list = []
    for stock in stocks:

        # load the 4 sources for the stock
        df_price = pd.read_csv('./data/%s_features_trading.csv' % stock)
        df_twitter = pd.read_csv('./data/%s_features_twitter.csv' % stock)
        df_minute_price = pd.read_csv('./data/%s_minute_price_features.csv' % stock)
        df_trend = pd.read_csv('./data/GOOGLE_TRENDS/%s/encoded_data/%s_features_google.csv' % (stock, stock))
        
        # merge sources together
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Date'],
                                                     how='inner'), [df_price, df_trend, df_twitter, df_minute_price])
        df_list.append(df_merged)


    df = pd.concat(df_list)

    # merge encoded moods together to get a general mood features dataframe
    mood_list_df = []
    for mood in ['debt', 'bloomberg', 'yahoo finance', 'buy stocks', 'sell stocks', 'VIX', 'stock risk', 'bullish_bearish']:
        mood_list_df.append(pd.read_csv('./data/GOOGLE_TRENDS/%s/encoded_data/%s_features_google.csv' % (mood, mood)))

    df_mood = reduce(lambda df1, df2: pd.merge(df1, df2, on='Date', how='inner'), mood_list_df)
    df_mood.to_csv('./data/GOOGLE_TRENDS/mood_features_g.csv', index=False)

    # add the google trends reflecting the mood of traders (common for all stocks)
    if os.path.exists('./data/GOOGLE_TRENDS/mood_features_g.csv'):
        df_mood_trend = pd.read_csv('./data/GOOGLE_TRENDS/mood_features_g.csv')
        df = df.merge(df_mood_trend, on='Date', how='inner')
        os.system('rm ./data/GOOGLE_TRENDS/mood_features_g.csv')

    df = df.sort_values('Date', ascending=False)

    return df
    
def wait_for_time(time_minute, source_name):
    
    now_minute = datetime.now().minute
    
    while now_minute < time_minute:
        print('Waiting. %s must be downloaded after HH:%s' % (source_name, time_minute))
        time.sleep(60)
        now_minute = datetime.now().minute



if __name__ == "__main__":
    main()

