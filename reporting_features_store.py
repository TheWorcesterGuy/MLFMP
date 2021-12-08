import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import os
pd.options.mode.chained_assignment = None
import glob
import datetime as dt
from functools import reduce


# CHECK FEATURE FILES
stocks = ['INTC', 'TSLA', 'AMZN', 'FB', 'AAPL', 'DIS', 'SPY', 'QQQ', 'GOOG', 'GOOGL', 'MSFT', 'NFLX', 'NVDA',
          'TWTR', 'AMD', 'WMT', 'JPM', 'BAC', 'PG']

df_list_stock = []
for stock in stocks:

    # check twitter features
    df_twitter_stock = pd.read_csv('./data/%s_features_twitter.csv' % stock)
    df_twitter_stock = df_twitter_stock[df_twitter_stock['Date'] == df_twitter_stock['Date'].max()]
    df_twitter_stock['source'] = 'twitter'
    df_twitter_stock['stock'] = stock
    df_twitter_stock = df_twitter_stock[['Date', 'source', 'stock']]

    # check google stock features
    try:
        if stock == 'FB':
            df_google_stock = pd.read_csv('./data/GOOGLE_TRENDS/facebook stock/encoded_data/facebook stock_features_google.csv')
        else:
            df_google_stock = pd.read_csv('./data/GOOGLE_TRENDS/%s/encoded_data/%s_features_google.csv' % (stock, stock))
        df_google_stock = df_google_stock[df_google_stock['Date'] == df_google_stock['Date'].max()]
        df_google_stock['source'] = 'google_stock'
        df_google_stock['stock'] = stock
        df_google_stock = df_google_stock[['Date', 'source', 'stock']]
    except:
        df_google_stock = pd.DataFrame([[np.nan, 'google stock', stock]], columns=['Date', 'source', 'stock'])
 

    # check google mood features
    moods = ['debt', 'bloomberg', 'yahoo finance', 'buy stocks', 'sell stocks', 'VIX', 'stock risk',
                         'investing.com', 'bullish_bearish']
    min_date = '3000-12-31'
    for mood in moods:
        df_google_mood = pd.read_csv('./data/GOOGLE_TRENDS/%s/encoded_data/%s_features_google.csv' % (mood, mood))
        df_google_mood = df_google_mood[df_google_mood['Date'] == df_google_mood['Date'].max()]
        if df_google_mood['Date'].iloc[0] < min_date:
            min_date = df_google_mood['Date'].iloc[0]
            df_google_mood = df_google_mood

    df_google_mood['source'] = 'google_mood'
    df_google_mood['stock'] = stock
    df_google_mood = df_google_mood[['Date', 'source', 'stock']]


    # check price features
    df_price_stock = pd.read_csv('./data/%s_features_trading.csv' % stock)
    df_price_stock = df_price_stock[df_price_stock['Date'] == df_price_stock['Date'].max()]
    df_price_stock['source'] = 'price'
    df_price_stock['stock'] = stock
    df_price_stock = df_price_stock[['Date', 'source', 'stock']]

    # check minute price data
    df_minute_price_stock = pd.read_csv('./data/%s_minute_price_features.csv' % stock)
    df_minute_price_stock = df_minute_price_stock[df_minute_price_stock['Date'] == df_minute_price_stock['Date'].max()]
    df_minute_price_stock['source'] = 'minute price'
    df_minute_price_stock['stock'] = stock
    df_minute_price_stock = df_minute_price_stock[['Date', 'source', 'stock']]

    # merge together
    df_stock = pd.concat([df_twitter_stock, df_google_stock, df_google_mood, 
                                                            df_price_stock, df_minute_price_stock], axis=0)[['stock', 'source', 'Date']]
    df_list_stock.append(df_stock)

df_stock_all = pd.concat(df_list_stock, axis=0)
df_stock_all.to_csv('./log/features_store/features_report.csv', index=False)



# CHECK ON MINUTE DATA
stocks = ['INTC', 'TSLA', 'AMZN', 'FB', 'AAPL', 'DIS', 'SPY', 'QQQ', 'GOOG', 'GOOGL', 'MSFT', 'NFLX', 'NVDA',
          'TWTR', 'AMD', 'WMT', 'JPM', 'BAC', 'PG']

df_minute_list = []
for stock in stocks:
    df_temp = pd.read_csv('./data/TRADE_DATA/minute_data/%s.csv' % stock)

    df_temp['stock'] = stock
    df_temp['Datetime'] = pd.to_datetime(df_temp['Datetime'])
    df_temp['date'] = df_temp['Datetime'].dt.date
    df_temp['time (UTC)'] = df_temp['Datetime'].dt.time
    df_temp = df_temp[['stock', 'date', 'time (UTC)', 'open', 'close']].tail(1)

    df_minute_list.append(df_temp)

df_minute_report = pd.concat(df_minute_list, axis=0)
df_minute_report = df_minute_report.reset_index(drop=True)

df_minute_report.to_csv('./log/features_store/minute_data.csv', index=False)



# CHECK ON PRICE DATA
stocks = ['INTC', 'TSLA', 'AMZN', 'FB', 'AAPL', 'DIS', 'SPY', 'QQQ', 'GOOG', 'GOOGL', 'MSFT', 'NFLX', 'NVDA',
          'TWTR', 'AMD', 'WMT', 'JPM', 'BAC', 'PG']

df_price_list = []
for stock in stocks:
    df_temp = pd.read_csv('./data/TRADE_DATA/price_data/%s.csv' % stock)

    df_temp['stock'] = stock
    df_temp = df_temp[['stock', 'Date', 'Open', 'Close']].tail(1)

    df_price_list.append(df_temp)

df_price_report = pd.concat(df_price_list, axis=0)
df_price_report = df_price_report.reset_index(drop=True)

df_price_report.to_csv('./log/features_store/price_data.csv', index=False)



# CHECK ON THE GOOGLE TRENDS DATA
google_trends_dir = ['facebook stock', 'SPY', 'AMD', 'AAPL', 'AMZN', 'QQQ', 'TSLA', 'MSFT',
                 'INTC', 'DIS', 'JPM', 'WMT', 'NFLX', 'GOOG', 'GOOGL', 'NVDA', 'TWTR',
                 'debt', 'bloomberg', 'yahoo finance', 'buy stocks', 'sell stocks', 'VIX', 'stock risk',
                     'bullish_bearish', 'investing.com']


df_last_google_trends = []
for google_trend in google_trends_dir:
    last_file = None
    files = glob.glob('./data/GOOGLE_TRENDS/%s/*.csv' % google_trend)
    for file in files:
        if str(datetime.today().month) in file and str(datetime.today().year) in file:
            last_file = file

    df_last_google = pd.read_csv(last_file)
    df_last_google = df_last_google.sort_values('date').tail(1)
    df_last_google['stock'] = df_last_google.columns[1]
    df_last_google['value'] = df_last_google[df_last_google.columns[1]]

    df_last_google = df_last_google[['stock', 'date', 'value', 'isPartial']]
    df_last_google = df_last_google.rename(columns={'date': 'last_datetime_NY'})
    df_last_google_trends.append(df_last_google)

df = pd.concat(df_last_google_trends, axis=0)
df.to_csv('./log/features_store/google_trend_data.csv', index=False)



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
    df_last_twitter = df_last_twitter[['stock', 'Datetime', 'compound']]
    df_last_twitter = df_last_twitter.rename(columns={'Datetime': 'last_datetime_utc'})

    df_last_twitters.append(df_last_twitter)

df = pd.concat(df_last_twitters, axis=0)
df.to_csv('./log/features_store/twitter_data.csv', index=False)


# REPORT ON THE NANS VARIABLES
df = pd.read_csv('./data/features_store.csv')

df = df[df['Date'] == df['Date'].max()]
max_date = df['Date'].max()

stocks = df['stock'].unique().tolist()

my_file = Path("./log/features_store/nans_per_stock.txt")
if my_file.is_file():
    os.system('rm ./log/features_store/nans_per_stock.txt')

with open('./log/features_store/nans_per_stock.txt', 'a') as f:
    f.write("Last date : %s\n\n\n" % max_date)

for stock in stocks:
    df_stock = df[df['stock'] == stock]
    nan_vals = df_stock.columns[df_stock.isna().any()].tolist()
    with open('./log/features_store/nans_per_stock.txt', 'a') as f:
        f.write("%s\n" % stock)
        for item in nan_vals:
            f.write("%s\n" % item)
        f.write("\n\n\n")




# REPORT ON THE FEATURE STORE NUMBER OF NANS
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
df_sample = df.iloc[:5000] # nb days x nb stock
df_metric_sample = df_sample[['Date', 'stock']]
df_metric_sample['mean_nb_nans'] = df_sample.isnull().sum(axis=1)
df_metric_sample['mean_percentage_nans'] = np.round(df_metric_sample['mean_nb_nans'] / df_sample.shape[1] * 100, 2)

df_metric_sample = df_metric_sample.groupby('stock').mean()
df_metric_sample['mean_nb_nans'] = np.round(df_metric_sample['mean_nb_nans'], 2)
df_metric_sample['mean_percentage_nans'] = np.round(df_metric_sample['mean_percentage_nans'], 2)

# join and save
df = df_metric.merge(df_metric_sample, on='stock', how='outer')

# write report in log
df.to_csv('./log/features_store/features_store_log.csv', index=False)


"""# APPLY HEALTH CHECK
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

    df_top_50 = df_top_features.merge(df_sample_top_features, on='stock', how='outer')
    df_top_50 = df_top_50[['stock', 'nb_nans_top_50', 'mean_nb_nans_top_50']]

    nb_missing_stocks = df_top_50['nb_nans_top_50'].isnull().sum()
    if nb_missing_stocks > 0:
        raise ValueError("%s stocks are missing in the last update" % nb_missing_stocks)

    df = df_top_50.merge(df, on='stock', how='outer')[['Date', 'nb_nans_top_50', 'mean_nb_nans_top_50',
                                                       'nb_nans', 'mean_nb_nans', 'percentage_nans', 'mean_percentage_nans']]

    if df['nb_nans_top_50'].sum() > 0:
        healthy_feature_store = False"""


