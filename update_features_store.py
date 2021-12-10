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


def main():

    update_start_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M')
    print('UPDATE START DATE (UTC) : %s\n\n' % update_start_date)

    stocks = ['INTC', 'TSLA',  'AMZN', 'FB', 'AAPL', 'DIS', 'SPY', 'QQQ', 'GOOG', 'GOOGL', 'MSFT', 'NFLX', 'NVDA',
              'TWTR', 'AMD', 'WMT', 'JPM', 'BAC', 'PG']

    google_trends_dir = ['facebook stock', 'SPY', 'AMD', 'AAPL', 'AMZN', 'QQQ', 'TSLA', 'MSFT',
                     'INTC', 'DIS', 'JPM', 'WMT', 'NFLX', 'GOOG', 'GOOGL', 'NVDA', 'TWTR',
                     'debt', 'bloomberg', 'yahoo finance', 'buy stocks', 'sell stocks', 'VIX', 'stock risk',
                         'investing.com', 'bullish_bearish']

    start = datetime.now()
    t0 = time.time()
    
    # update minute price data download and create features
    os.system('python download_minute_price.py')
    apply_parallel_command(5, "./create_minute_price_features.py", stocks)
    
    stop_high_res_price = datetime.now()

    # update google trends download and create features
    os.system("python download_google_trends.py")
    apply_parallel_command(10, "./create_google_trends_features.py", google_trends_dir)

    stop_google_trends = datetime.now()

    # update download price data and create features
    os.system("python download_price.py")
    apply_parallel_command(5, "./create_trading_features.py", stocks)

    stop_trade = datetime.now()

    # update tweet download and create features
    apply_parallel_command(6, "./download_tweets.py", stocks)
    apply_parallel_command(6, "./encode_tweets.py", stocks)
    apply_parallel_command(3, "./create_twitter_features.py", stocks)
    

    # merge the two sources of features
    df = merge_files(stocks)
    df = df.set_index(['Date', 'stock'])
    df.to_csv('./data/features_store.csv')
    print(df)
    print('features store updated')

    #os.system("rm ./data/*features_trading.csv")
    #os.system("rm ./data/*features_twitter.csv")

    stop = datetime.now()
    
    print('\n Time to update minute price features: ', (stop_high_res_price - start))
    print('\n Time to update google trend features: ', (stop_google_trends - start))
    print('\n Time to update trade features: ', (stop_trade - stop_google_trends))
    print('\n Time to update twitter features: ', (stop - stop_trade))
    print('\n Time to update features: ', (stop - start))
    print('\nLast row added:', df.head(1))
    
    
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
        df_price = pd.read_csv('./data/%s_features_trading.csv' % stock)
        df_twitter = pd.read_csv('./data/%s_features_twitter.csv' % stock)
        df_minute_price = pd.read_csv('./data/%s_minute_price_features.csv' % stock)

        # get matching google stock keyword for a given stock name
        if stock == 'FB':
            google_keyword = 'facebook stock'
        else:
            google_keyword = stock

        # we don't have google trends for all stocks yet
        if os.path.exists('./data/GOOGLE_TRENDS/%s/encoded_data/%s_features_google.csv' % (google_keyword,
                                                                                           google_keyword)):
            df_trend = pd.read_csv('./data/GOOGLE_TRENDS/%s/encoded_data/%s_features_google.csv' % (google_keyword,
                                                                                                    google_keyword))
            df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Date'],
                                                     how='inner'), [df_price, df_twitter, df_minute_price, df_trend])
            df_list.append(df_merged)
            #os.system(" rm './data/GOOGLE_TRENDS/%s/encoded_data/%s_features_google.csv'" % (stock, stock))
        else:
            df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Date'],
                                                            how='inner'), [df_price, df_twitter, df_minute_price])
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
        
    #os.system('rm ./data/*_minute_price_features.csv')

    df = df.sort_values('Date', ascending=False)

    return df


if __name__ == "__main__":
    main()

