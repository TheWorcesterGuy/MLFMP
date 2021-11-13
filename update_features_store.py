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


    stocks = ['INTC', 'TSLA',  'AMZN', 'FB', 'AAPL', 'DIS', 'SPY', 'QQQ', 'GOOG', 'GOOGL', 'MSFT', 'NFLX', 'NVDA', 'BA',
              'TWTR', 'AMD', 'WMT', 'JPM', 'BAC', 'JNJ', 'PG', 'NKE']

    google_trends = ['facebook stock', 'SPY', 'AMD', 'AAPL', 'AMZN', 'QQQ', 'TSLA', 'MSFT', 'boeing stock',
                     'INTC', 'DIS', 'JPM', 'WMT', 'NFLX', 'GOOG', 'GOOGL', 'NVDA', 'TWTR',
                     'debt', 'bloomberg', 'yahoo finance', 'buy stocks', 'sell stocks', 'VIX', 'stock risk']

    start = datetime.now()
    t0 = time.time()

    # update google trends download
    os.system("python download_google_trends.py")

    # create google trend features
    apply_parallel_command(10, "./create_google_trends_features.py", google_trends)

    stop_google_trends = datetime.now()

    # get trading features
    os.system("python download_price.py")
    apply_parallel_command(5, "./create_trading_features.py", stocks)

    stop_trade = datetime.now()

    # update tweet downloading and encoding
    apply_parallel_command(6, "./download_tweets.py", stocks)
    apply_parallel_command(6, "./encode_tweets.py", stocks)


    # get twitter features
    apply_parallel_command(3, "./create_twitter_features.py", stocks)

    # merge the two sources of features
    df = merge_files(stocks)
    df = df.set_index(['Date', 'stock'])
    df.to_csv('./data/features_store.csv')
    print(df)
    print('features store updated')

    os.system("rm ./data/*features_trading.csv")
    os.system("rm ./data/*features_twitter.csv")

    stop = datetime.now()
    print('\n Time to update google trend features: ', (stop_google_trends - start))
    print('\n Time to update trade features: ', (stop_trade - stop_google_trends))
    print('\n Time to update twitter features: ', (stop - stop_trade))
    print('\n Time to update features: ', (stop - start))


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
        if os.path.exists('./data/GOOGLE_TRENDS/%s/encoded_data/%s_features_google.csv' % (stock, stock)): # we don't have google trends for all stocks yet
            df_trend = pd.read_csv('./data/GOOGLE_TRENDS/%s/encoded_data/%s_features_google.csv' % (stock, stock))
            df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Date'],
                                                            how='inner'), [df_price, df_twitter, df_trend])
            df_list.append(df_merged)
            #os.system(" rm './data/GOOGLE_TRENDS/%s/encoded_data/%s_features_google.csv'" % (stock, stock))
        else:
            df_merged = df_price.merge(df_twitter, on='Date', how='inner')
            df_list.append(df_merged)

    df = pd.concat(df_list)

    # merge encoded moods together to get a general mood features dataframe
    mood_list_df = []
    for mood in ['debt', 'bloomberg', 'yahoo finance', 'buy stocks', 'sell stocks', 'VIX', 'stock risk']:
        mood_list_df.append(pd.read_csv('./data/GOOGLE_TRENDS/%s/encoded_data/%s_features_google.csv' % (mood, mood)))

    df_mood = reduce(lambda df1, df2: pd.merge(df1, df2, on='Date', how='inner'), mood_list_df)
    df_mood.to_csv('./data/GOOGLE_TRENDS/mood_features_g.csv', index=False)

    # add the google trends reflecting the mood of traders (common for all stocks)
    if os.path.exists('./data/GOOGLE_TRENDS/mood_features_g.csv'):
        df_mood_trend = pd.read_csv('./data/GOOGLE_TRENDS/mood_features_g.csv')
        df = df.merge(df_mood_trend, on='Date', how='inner')
        #os.system('rm /data/GOOGLE_TRENDS/mood_features_g.csv')

    df = df.sort_values('Date', ascending=False)

    return df


if __name__ == "__main__":
    main()

