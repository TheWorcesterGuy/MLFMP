#!/usr/bin/env python3
import datetime

import yfinance as yf
from yahoofinancials import YahooFinancials
import pandas as pd
import numpy as np
import glob
import os
import sys
import datetime as dt
import warnings
import pytz
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


stock = sys.argv[1]


def main():

    print('Creating Twitter features for %s...' % stock)
    # load yahoo stock prices
    df_stock = yf.download(stock, start='2016-01-01', end='3000-12-31', progress=False).reset_index(drop=False)[['Date', 'Close', 'Open']]
    df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')

    # load Twitter data
    list_files = [pd.read_csv(file, lineterminator='\n')[['Datetime', 'number_cashtag', 'nbFollowers',
                                                          'compound', 'LM_score']] for file in
                  glob.glob("data/TWITTER_DATA/%s/encoded_data/*encoded*.csv" % stock)]
    df = pd.concat(list_files, axis=0)

    # convert UTC time to NY time and save tweets after 8:35 NY time for next day
    df = fix_tweet_timing(df, df_stock)

    # aggregate tweets
    df = aggregate_tweets(df)

    # add stock market prices
    df = df.merge(df_stock, on='Date', how='outer')
    
    # create stock market name column
    df['stock'] = stock

    # create derived features
    df = create_derived_features(df)

    # make deltas (not to be used in training!)
    df['delta'] = (df['Close'] - df['Open']) / df['Open'] * 100   # open at 9:30 am NY and close at 17:00 pm NY
    df['delta_class'] = np.nan

    df.loc[df['delta'] > 0, 'delta_class'] = 1
    df.loc[df['delta'] <= 0, 'delta_class'] = -1

    df = df.sort_values('Date', ascending=False)

    df = df[df['Date'] >= pd.to_datetime('2016-02-15', format='%Y-%m-%d')]
    df = df.fillna(0)

    df.to_csv('./data/%s_features_twitter.csv' % stock, index=False)


def compute_vader_sent(df):
    Mpos = df['compound'][df['compound'] > 0.1].count() * (df['nbFollowers'][df['compound'] > 0.1].count()) ** 0.5
    Mneg = df['compound'][df['compound'] < -0.1].count() * (df['nbFollowers'][df['compound'] < -0.1].count()) ** 0.5
    if Mneg + Mpos == 0:
        return - 1
    else:
        score = (Mpos - Mneg) / (Mpos + Mneg)
        #score = np.log((1 + Mpos) / (1 + Mneg))
        return score


def compute_LM_sent(df):
    Mneg = df['LM_score'][df['LM_score'] < 0].count() * (df['nbFollowers'][df['LM_score'] < 0].count()) ** 0.5
    Mpos = df['LM_score'][df['LM_score'] > 0].count() * (df['nbFollowers'][df['LM_score'] > 0].count()) ** 0.5

    if Mneg + Mpos == 0:
        return - 1
    else:
        score = (Mpos - Mneg) / (Mpos + Mneg)
        #score = np.log((1 + Mpos) / (1 + Mneg))
        return score


def nb_tweets(df):
    nb = df['compound'].count()
    return nb


def std_vader(df):
    std = df['compound'].std()
    return std


def std_LM(df):
    std = df['LM_score'].std()
    return std


def fix_tweet_timing(df, df_stock):

    # convert df to NY time
    eastern = pytz.timezone('US/Eastern')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Datetime'] = df['Datetime'].dt.tz_convert(eastern)

    # drop days of tweet if no stock except if last day
    df['Date'] = pd.to_datetime(df['Datetime'].dt.date)
    df = df.merge(df_stock, on='Date', how='left')
    df = df[(df['Open'].notna()) | (df['Date'] == df['Date'].max())]
    df = df.drop(['Date', 'Close', 'Open'], axis=1)

    # tweets after 8:35 am NY time are saved for next day which means we shift forward by 15 hours and 25 minutes
    df['Datetime'] = df['Datetime'] + pd.DateOffset(hours=15, minutes=25)

    # tweets that fall on saturday must be redirected for monday > adding 48 hours to the tweets on saturday
    df['dayweek'] = df['Datetime'].dt.dayofweek
    df.loc[df['dayweek'] == 5, 'Datetime'] += pd.Timedelta(hours=48)
    df['Date'] = pd.to_datetime(df['Datetime'].dt.date, errors='coerce')
    df = df.drop(['dayweek'], axis=1)

    return df


def aggregate_tweets(df):

    df1 = df[df['number_cashtag'] == 1]
    df2 = df[df['number_cashtag'] >= 1]

    # vader features
    df_a1 = df1.groupby('Date', as_index=False).apply(lambda group: compute_vader_sent(group))
    df_a1['Date'] = pd.to_datetime(df_a1['Date'], errors='coerce')
    df_a1.columns = ['Date', 'vader_1$']

    df_a2 = df2.groupby('Date', as_index=False).apply(lambda group: compute_vader_sent(group))
    df_a2['Date'] = pd.to_datetime(df_a2['Date'], errors='coerce')
    df_a2.columns = ['Date', 'vader_2$']

    # LM features
    df_b1 = df1.groupby('Date', as_index=False).apply(lambda group: compute_LM_sent(group))
    df_b1['Date'] = pd.to_datetime(df_b1['Date'], errors='coerce')
    df_b1.columns = ['Date', 'LM_1$']

    df_b2 = df2.groupby('Date', as_index=False).apply(lambda group: compute_LM_sent(group))
    df_b2['Date'] = pd.to_datetime(df_b2['Date'], errors='coerce')
    df_b2.columns = ['Date', 'LM_2$']

    # Tweet number features (normalize monthly)
    df_c1 = df1.groupby('Date', as_index=False).apply(lambda group: nb_tweets(group))
    df_c1['Date'] = pd.to_datetime(df_c1['Date'], errors='coerce')
    df_c1.columns = ['Date', 'nb_tweet_1$']

    df_c2 = df2.groupby('Date', as_index=False).apply(lambda group: nb_tweets(group))
    df_c2['Date'] = pd.to_datetime(df_c2['Date'], errors='coerce')
    df_c2.columns = ['Date', 'nb_tweet_2$']

    # Std Vader features
    df_d1 = df1.groupby('Date', as_index=False).apply(lambda group: std_vader(group))
    df_d1['Date'] = pd.to_datetime(df_d1['Date'], errors='coerce')
    df_d1.columns = ['Date', 'std_vader_1$']

    df_d2 = df2.groupby('Date', as_index=False).apply(lambda group: std_vader(group))
    df_d2['Date'] = pd.to_datetime(df_d2['Date'], errors='coerce')
    df_d2.columns = ['Date', 'std_vader_2$']

    # Std LM features
    df_e1 = df1.groupby('Date', as_index=False).apply(lambda group: std_LM(group))
    df_e1['Date'] = pd.to_datetime(df_e1['Date'], errors='coerce')
    df_e1.columns = ['Date', 'std_LM_1$']

    df_e2 = df2.groupby('Date', as_index=False).apply(lambda group: std_LM(group))
    df_e2['Date'] = pd.to_datetime(df_e2['Date'], errors='coerce')
    df_e2.columns = ['Date', 'std_LM_2$']

    df = df_a1.merge(df_a2, on='Date', how='outer').merge(df_b1, on='Date', how='outer').merge(df_b2, on='Date', how='outer')\
        .merge(df_c1, on='Date', how='outer').merge(df_c2, on='Date', how='outer').merge(df_d1, on='Date', how='outer')\
        .merge(df_d2, on='Date', how='outer').merge(df_e1, on='Date', how='outer').merge(df_e2, on='Date', how='outer')

    return df


def create_derived_features(df):

    raw_features = ['LM_1$', 'vader_1$', 'std_vader_1$', 'std_LM_1$', 'nb_tweet_1$', 'LM_2$', 'vader_2$',
                    'std_vader_2$', 'std_LM_2$', 'nb_tweet_2$']
                    
    for feature in raw_features:

        df[feature + 'lag1'] = df[feature].shift(1)
        df[feature + 'lag2'] = df[feature].shift(2)
        df[feature + 'lag3'] = df[feature].shift(3)
        df[feature + 'lag4'] = (df[feature].shift(4) + df[feature].shift(5)) / 2
        df[feature + 'lag6'] = (df[feature].shift(6) + df[feature].shift(7)) / 2
        df[feature + 'lag8'] = (df[feature].shift(8) + df[feature].shift(9)) / 2
        df[feature + 'lag10'] = (df[feature].shift(10) + df[feature].shift(11)) / 2
        df[feature + 'lag12'] = (df[feature].shift(12) + df[feature].shift(13)) / 2
        df[feature + 'lag14'] = (df[feature].shift(14) + df[feature].shift(14)) / 2
        df[feature + 'lag20'] = (df[feature].shift(15) + df[feature].shift(16) + df[feature].shift(17) + df[feature].shift(18) + df[feature].shift(19) + df[feature].shift(20)) / 6
        
        df[feature + 'delta1'] = df[feature + 'lag1'] / df[feature]
        df[feature + 'delta2'] = df[feature + 'lag2'] / df[feature]
        df[feature + 'delta3'] = df[feature + 'lag3'] / df[feature]
        df[feature + 'delta4'] = df[feature + 'lag4'] / df[feature]
        df[feature + 'delta6'] = df[feature + 'lag6'] / df[feature]
        df[feature + 'delta8'] = df[feature + 'lag8'] / df[feature]
        df[feature + 'delta10'] = df[feature + 'lag10'] / df[feature]
        df[feature + 'delta12'] = df[feature + 'lag12'] / df[feature]
        df[feature + 'delta14'] = df[feature + 'lag14'] / df[feature]
        df[feature + 'delta20'] = df[feature + 'lag20'] / df[feature]
        
        df[feature + 'delta_mean3'] = df[[feature + 'delta1', feature + 'delta2', feature + 'delta3']].mean(axis=1) / 3
        df[feature + 'delta_mean6'] = df[[feature + 'delta4', feature + 'delta6']].mean(axis=1) / 3
        df[feature + 'delta_mean12'] = df[[feature + 'delta6', feature + 'delta8', feature + 'delta10', feature + 'delta12']].mean(axis=1) / 5
        
        df[feature + 'dev_delta21'] = (df[feature + 'delta1'] / df[feature + 'delta2'])
        df[feature + 'dev_delta31'] = (df[feature + 'delta1'] / df[feature + 'delta3'])
        df[feature + 'dev_delta41'] = (df[feature + 'delta1'] / df[feature + 'delta4'])
        df[feature + 'dev_delta51'] = (df[feature + 'delta1'] / df[feature + 'delta6'])
        
        df[feature + 'dev_delta31_smooth'] = (df[feature + 'delta1'] + df[feature + 'delta2']) / (df[feature + 'delta3'] + df[feature + 'delta4'])
        df[feature + 'dev_delta41_smooth'] = (df[feature + 'delta1'] + df[feature + 'delta2']) / (df[feature + 'delta6'])
        df[feature + 'dev_delta51_smooth'] = (df[feature + 'delta1'] + df[feature + 'delta2']) / (df[feature + 'delta8'])
         
        df[feature + 'dev_delta32'] = (df[feature + 'delta2'] + df[feature + 'delta3']) / df[feature + 'delta4']
        df[feature + 'dev_delta42'] = (df[feature + 'delta2'] + df[feature + 'delta3']) / df[feature + 'delta6']
        df[feature + 'dev_delta52'] = (df[feature + 'delta2'] + df[feature + 'delta3']) / df[feature + 'delta8']

        if 'std' not in feature:
            df[feature + 'std2'] = df[[feature, feature + 'lag1', feature + 'lag2']].std(axis=1)
            df[feature + 'std6'] = df[[feature + 'lag3', feature + 'lag4', feature + 'lag6']].std(axis=1)
        
        if 'std' not in feature and 'nb_tweet' not in feature:
            df[feature + 'mean_3'] = (df[feature + 'lag1'] + df[feature + 'lag2'] + df[feature + 'lag3']) / 3
            df[feature + 'mean_6'] = (df[feature + 'lag1'] + df[feature + 'lag2'] + df[feature + 'lag3']
                                              + df[feature + 'lag4'] + df[feature + 'lag6']) / 6
            df[feature + 'mean_12'] = (df[feature + 'lag8'] + df[feature + 'lag10'] + df[feature + 'lag12']) / 6

        if 'nb_tweet' in feature:
  
            del df[feature + 'lag1']
            del df[feature + 'lag2']
            del df[feature + 'lag3']
            del df[feature + 'lag4']
            del df[feature + 'lag6']
            del df[feature + 'lag8']
            del df[feature + 'lag10']
            del df[feature + 'lag12']
            del df[feature + 'lag14']
            del df[feature + 'lag20']

    df['delta_stock1'] = (df['Close'].shift(1) - df['Open'].shift(1)) / df['Open'].shift(1)
    df['delta_stock2'] = (df['Close'].shift(2) - df['Open'].shift(2)) / df['Open'].shift(2)
    df['delta_stock3'] = (df['Close'].shift(3) - df['Open'].shift(3)) / df['Open'].shift(3)
    df['delta_stock4'] = (df['Close'].shift(4) - df['Open'].shift(4)) / df['Open'].shift(4)
    df['delta_stock5'] = (df['Close'].shift(5) - df['Open'].shift(5)) / df['Open'].shift(5)
    df['delta_stock6'] = (df['Close'].shift(6) - df['Open'].shift(6)) / df['Open'].shift(6)
    df['mean_delta_stock3'] = df[['delta_stock1', 'delta_stock2', 'delta_stock3']].mean(axis=1)
    df['mean_delta_stock6'] = df[['delta_stock4', 'delta_stock5', 'delta_stock6']].mean(axis=1)

    df['delta_dev_stock2'] = df['delta_stock1'] - df['delta_stock2']
    df['delta_dev_stock3'] = df['delta_stock2'] - df['delta_stock3']
    df['delta_dev_stock4'] = df['delta_stock3'] - df['delta_stock4']
    df['delta_dev_stock5'] = df['delta_stock4'] - df['delta_stock5']
    df['delta_dev_stock6'] = df['delta_stock5'] - df['delta_stock6']
    
    print(df.tail(1))

    return df


if __name__ == "__main__":
    main()
