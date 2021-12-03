import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import os
pd.options.mode.chained_assignment = None
import glob
import datetime as dt


# CHECK ON THE GOOGLE TRENDS DATA
google_trends_dir = ['facebook stock', 'SPY', 'AMD', 'AAPL', 'AMZN', 'QQQ', 'TSLA', 'MSFT',
                 'INTC', 'DIS', 'JPM', 'WMT', 'NFLX', 'GOOG', 'GOOGL', 'NVDA', 'TWTR',
                 'debt', 'bloomberg', 'yahoo finance', 'buy stocks', 'sell stocks', 'VIX', 'stock risk',
                     'bullish_bearish', 'investing.com']


df_last_google_trends = []
for google_trend in google_trends_dir:
    last_file = None
    print(google_trend)
    files = glob.glob('./data/GOOGLE_TRENDS/%s/*.csv' % google_trend)
    for file in files:
        if str(datetime.today().month) in file and str(datetime.today().year) in file:
            last_file = file

    df_last_google = pd.read_csv(last_file)
    df_last_google = df_last_google.sort_values('date').tail(1)
    df_last_google['stock'] = df_last_google.columns[1]
    df_last_google['value'] = df_last_google[df_last_google.columns[1]]

    df_last_google = df_last_google[['date', 'stock', 'value', 'isPartial']]
    df_last_google = df_last_google.rename(columns={'date': 'last_datetime'})
    df_last_google_trends.append(df_last_google)

df = pd.concat(df_last_google_trends, axis=0)
df.to_csv('./log/google_trend_data_check.csv', index=False)



# CHECK ON TWITTER DATA

stocks = ['INTC', 'TSLA', 'AMZN', 'FB', 'AAPL', 'DIS', 'SPY', 'QQQ', 'GOOG', 'GOOGL', 'MSFT', 'NFLX', 'NVDA',
          'TWTR', 'AMD', 'WMT', 'JPM', 'BAC', 'PG']

df_last_twitters = []
for stock in stocks:
    files = glob.glob('./data/TWITTER_DATA/%s/encoded_data/*.csv' % stock)
    for file in files:
        if str(datetime.today().isocalendar()[1]) in file and str(datetime.today().year) in file:
            last_file = file
    df_last_twitter = pd.read_csv(last_file)
    df_last_twitter = df_last_twitter.sort_values('Datetime').dropna() #### to be fixed

    df_last_twitter = df_last_twitter.tail(1)
    df_last_twitter['stock'] = stock
    df_last_twitter = df_last_twitter[['Datetime', 'stock', 'compound']]
    df_last_twitter = df_last_twitter.rename(columns={'Datetime': 'last_datetime_utc'})

    df_last_twitters.append(df_last_twitter)

df = pd.concat(df_last_twitters, axis=0)
df.to_csv('./log/twitter_data_check.csv', index=False)



# CHECK ON THE FEATURE STORE
current_date = datetime.today().strftime('%Y-%m-%d')

df = pd.read_csv('./data/features_store.csv')
healthy_feature_store = True

# take latest rows
df_last_day = df[df['Date'] == df['Date'].max()]


# compute metrics for latest rows
df_metric = df_last_day[['Date', 'stock']]
df_metric['nb_nans'] = df_last_day.isnull().sum(axis=1)

df_metric['percentage_nans'] = np.round(df_metric['nb_nans'] / df_last_day.shape[1] * 100, 2)

# compute global metric to compare against
df_sample = df.iloc[-500:]
df_metric_sample = df_sample[['Date', 'stock']]
df_metric_sample['mean_nb_nans'] = df_sample.isnull().sum(axis=1)
df_metric_sample['mean_percentage_nans'] = np.round(df_metric_sample['mean_nb_nans'] / df_sample.shape[1] * 100, 2)

df_metric_sample = df_metric_sample.groupby('stock').mean()
df_metric_sample['mean_nb_nans'] = np.round(df_metric_sample['mean_nb_nans'], 2)
df_metric_sample['mean_percentage_nans'] = np.round(df_metric_sample['mean_percentage_nans'], 2)

# join and save
df = df_metric.merge(df_metric_sample, on='stock', how='outer')

# write report in log
df.to_csv('./log/features_store_log.csv', index=False)


my_file = Path("./data/top_50.csv")
if my_file.is_file():
    df_features = pd.read_csv('./data/top_50.csv')
    top_features = ['stock'] + df_features['Features'].values.tolist()

    df_top_features = df_last_day[top_features]
    df_sample_top_features = df_sample[top_features]

    df_top_features['nb_nans_top_50'] = df_top_features.isnull().sum(axis=1)

    df_sample_top_features['mean_nb_nans_top_50'] = df_sample_top_features.isnull().sum(axis=1)
    df_sample_top_features = df_sample_top_features[['stock', 'mean_nb_nans_top_50']].groupby('stock').mean()
    df_sample_top_features['mean_nb_nans_top_50'] = np.round(df_sample_top_features['mean_nb_nans_top_50'], 2)
    print(df_sample_top_features)
    print(df_top_features)

    df_top_50 = df_top_features.merge(df_sample_top_features, on='stock', how='outer')
    df_top_50 = df_top_50[['stock', 'nb_nans_top_50', 'mean_nb_nans_top_50']]

    nb_missing_stocks = df_top_50['nb_nans_top_50'].isnull().sum()
    if nb_missing_stocks > 0:
        raise ValueError("%s stocks are missing in the last update" % nb_missing_stocks)

    df = df_top_50.merge(df, on='stock', how='outer')[['Date', 'nb_nans_top_50', 'mean_nb_nans_top_50',
                                                       'nb_nans', 'mean_nb_nans', 'percentage_nans', 'mean_percentage_nans']]

    if df['nb_nans_top_50'].sum() > 0:
        healthy_feature_store = False


# if last date in feature store is not today's
if df['Date'].max() != current_date:
    healthy_feature_store = False

# delete feature store if healthy check failed
#if not healthy_feature_store :
    #os.system('rm ./data/features_store.csv')
