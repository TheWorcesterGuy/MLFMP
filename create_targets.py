import alpaca_trade_api as tradeapi
from datetime import datetime
from alpaca_trade_api.rest import TimeFrame
import time
from datetime import datetime, timedelta
import pytz
import os
import pandas as pd
import numpy as np
import warnings
#from email_updates_error import *
import yfinance as yf
import sys
import time
import finnhub
import glob
import random
from functools import reduce


def main():

        # df = pd.read_csv('./data/features_store_dev.csv')
        # df = df[df['stock'] == 'SPY']

        # print(df)
        # for col in df.columns:
        #         print(col)


        # df = df.drop(['Date', 'delta', 'target_11h', 'target_10h', 'target_15h', 'target_13h', 'target_14h', 'target_12h', 'stock'], axis=1)
        
        # df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # df = df.dropna()

        # y = df['delta_class']
        # X = df.drop(['delta_class'], axis=1)

        # X_train = X.iloc[:-200]
        # y_train = y.iloc[:-200]

        # X_test = X.iloc[-200:]
        # y_test = y.iloc[-200:]

        # from sklearn.ensemble import RandomForestClassifier

        # model = RandomForestClassifier(max_depth=25, random_state=0, n_estimators=400)
        # model.fit(X_train, y_train)

        # from sklearn import metrics
        # from sklearn.metrics import accuracy_score

        # y_pred = model.predict(X_test)
        # print(accuracy_score(y_test, y_pred))

        # exit(0)

        # GET LABELS FOR ALL STOCKS
        stocks = ['INTC', 'TSLA', 'AMZN', 'FB', 'AAPL', 'DIS', 'SPY', 'QQQ', 'GOOG', 'GOOGL', 'MSFT', 'NFLX', 'NVDA',
        	        'TWTR', 'AMD', 'WMT', 'JPM', 'BAC', 'PG']
        df_list = []
        for stock in stocks:

                df = pd.read_csv('./data/TRADE_DATA/minute_data/%s.csv' % stock)
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df = df.set_index('Datetime')
                
                # Convert to NY time
                eastern = pytz.timezone('US/Eastern')
                df.index = df.index.tz_convert(eastern)
                df = df.sort_values('Datetime')
                df = df.reset_index()

                # get prices at stage 10am, 11am, Noon, 1pm, 2pm, 3pm
                df['Date'] = df['Datetime'].dt.date
                df['hour'] = df['Datetime'].dt.hour
                df['minute'] = df['Datetime'].dt.minute
                df = df[['Datetime', 'Date', 'hour', 'minute', 'open']]

                # get price at given hours
                df_10h = get_hour_stage_price(df, 10, 0)
                df_10h30 = get_hour_stage_price(df, 10, 30)
                df_11h = get_hour_stage_price(df, 11, 0)
                df_11h30 = get_hour_stage_price(df, 11, 30)
                df_12h = get_hour_stage_price(df, 12, 0)
                df_12h30 = get_hour_stage_price(df, 12, 30)
                df_13h = get_hour_stage_price(df, 13, 0)
                df_13h30 = get_hour_stage_price(df, 13, 30)
                df_14h = get_hour_stage_price(df, 14, 0)
                df_14h30 = get_hour_stage_price(df, 14, 30)
                df_15h = get_hour_stage_price(df, 15, 0)
                df_15h30 = get_hour_stage_price(df, 15, 30)
                df_16h = get_hour_stage_price(df, 16, 0)

                # get price at opening
                df_opening = df[(df['hour'] == 9) & (df['minute'] >= 30)]
                df_opening = df_opening.drop_duplicates(subset=['Date', 'hour'], keep='first')
                df_opening = df_opening[['Date', 'open']]
                df_opening = df_opening.rename(columns={'open':'opening'})

                # merge all price states
                df_targets = reduce(lambda df1, df2: pd.merge(df1, df2, on='Date'), \
                                        [df_opening, df_10h, df_10h30, df_11h, df_11h30, df_12h, df_12h30, \
                                        df_13h, df_13h30, df_14h, df_14h30, df_15h, df_15h30, df_16h])

                # create targets for several times
                df_targets['target_10h'] = df_targets['10h_price'] - df_targets['opening']
                df_targets['target_10h30'] = df_targets['10h30_price'] - df_targets['opening']
                df_targets['target_11h'] = df_targets['11h_price'] - df_targets['opening']
                df_targets['target_11h30'] = df_targets['11h30_price'] - df_targets['opening']
                df_targets['target_12h'] = df_targets['12h_price'] - df_targets['opening'] 
                df_targets['target_12h30'] = df_targets['12h30_price'] - df_targets['opening'] 
                df_targets['target_13h'] = df_targets['13h_price'] - df_targets['opening']
                df_targets['target_13h30'] = df_targets['13h30_price'] - df_targets['opening']
                df_targets['target_14h'] = df_targets['14h_price'] - df_targets['opening']
                df_targets['target_14h30'] = df_targets['14h30_price'] - df_targets['opening']
                df_targets['target_15h'] = df_targets['15h_price'] - df_targets['opening']
                df_targets['target_15h30'] = df_targets['15h30_price'] - df_targets['opening']
                df_targets['target_16h'] = df_targets['16h_price'] - df_targets['opening']

                
                df_targets = df_targets.drop(['opening', '10h_price', '10h30_price', '11h_price', '11h30_price', \
                                        '12h_price', '12h30_price', '13h_price', '13h30_price', '14h_price', '14h30_price', 
                                        '15h_price', '15h30_price', '16h_price'], axis=1)

                for target in ['target_10h', 'target_10h30', 'target_11h', 'target_11h30', 'target_12h', 'target_12h30', 'target_13h',
                                        'target_13h30', 'target_14h', 'target_14h30', 'target_15h', 'target_15h30', 'target_16h']:
                        df_targets[target] = df_targets.apply(lambda row: label_price(row, target), axis=1)

                # Shift to match price target for next day
                df_targets['target_10h'] = df_targets['target_10h'].shift(-1)
                df_targets['target_11h'] = df_targets['target_11h'].shift(-1)
                df_targets['target_12h'] = df_targets['target_12h'].shift(-1)
                df_targets['target_13h'] = df_targets['target_13h'].shift(-1)
                df_targets['target_14h'] = df_targets['target_14h'].shift(-1)
                df_targets['target_15h'] = df_targets['target_15h'].shift(-1)

                # add stock name
                df_targets['stock'] = stock

                # drop last day (nan - no outcome yet for current day)
                df_targets = df_targets.dropna()
                df_list.append(df_targets)

        df_targets = pd.concat(df_list, axis=0)
        df_targets['Date'] = pd.to_datetime(df_targets['Date'])


        # MERGE LABELS TO FEATURE STORE
        df_feature_store = pd.read_csv('./data/features_store.csv')
        df_feature_store['Date'] = pd.to_datetime(df_feature_store['Date'])

        df_feature_store = df_feature_store.merge(df_targets, on=['Date', 'stock'])
        df_feature_store = df_feature_store.drop(['delta_class', 'delta'], axis=1)
        df_feature_store.to_csv('./data/features_store_dev.csv', index=False)


def label_price(row, col):
        if row[col] >= 0 :
                return 1
        if row[col] < 0:
                return -1

def get_hour_stage_price(df, hour, minute):
        df_hour = df[df['hour'] == hour]
        df_hour = df_hour[df_hour['minute'] >= minute]
        df_hour = df_hour.drop_duplicates(subset=['Date'], keep='first')
        df_hour = df_hour[['Date', 'open']]
        if minute == 0:
                minute = ''
        df_hour = df_hour.rename(columns={'open':'%sh%s_price' % (hour, minute)})
        return df_hour

if __name__ == "__main__":
    main()


