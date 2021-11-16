#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Jun 30 23:35:28 2021

@author: christian
'''
import yfinance as yf
from fredapi import Fred
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime
from alpaca_trade_api.rest import TimeFrame
import warnings
import sys
import time
from scipy.fft import fft
import finnhub
import glob
import random
warnings.simplefilter(action='ignore')

def main():
    print('\n Creating price & econometric features for %s ...' % sys.argv[1])
    features(symbol = sys.argv[1]).execute()
    print(' Done creating price features for %s' % sys.argv[1])
    #features(symbol='AAPL').execute() # This line is reserved for testing

class features:
    def __init__(self, symbol):
        
        self.symbol = symbol
        
    def get_price(self, ticker):
        
        df = pd.read_csv('./data/TRADE_DATA/price_data/%s.csv' % ticker)
        
        return df
        
    def data_collect_price(self):
        
        today = datetime.today()

        World_tickers = ['^FCHI','^STOXX50E','^AEX','000001.SS','^HSI','^N225','^BSESN','^SSMI','^IBEX']
        Reinforce_tickers = ['^VVIX','^VIX','SPY', 'QQQ','GC=F','CL=F','SI=F','EURUSD=X','JPY=X',
                     'XLF','XLK','XLV','XLY', 'INTC', 'AMZN', 'FB', 'AAPL', 'DIS', 'TSLA', 'GOOG', 
                     'GOOGL', 'MSFT', 'NFLX', 'NVDA', 'BA', 'TWTR', 'AMD', 'WMT', 
                     'JPM',  'BAC', 'JNJ', 'PG', 'NKE']

        combinations = [0, 1, 2, 3, 5, 7, 10, 15, 20, 50]
        
        price_data = self.get_price(self.symbol)
        price_data.set_index('Date', inplace=True)
        price_data['symbol'] = self.symbol
        
        for wt in World_tickers:
            support = self.get_price(wt)
            support.set_index('Date', inplace=True)
            for k in combinations:
                price_data[wt + str(k)] = ((support['Open'].shift(-1) - support['Close'].shift(k)) / support['Close'].shift(k)) * 100
            price_data[wt + ' High ' + str(0)] = ((support['High'] - support['Low']) / support['Open']) * 100
        
        for rt in Reinforce_tickers:
            reinf = self.get_price(rt)
            reinf.set_index('Date', inplace=True)
            for j in combinations:
                price_data[rt + ' - ' + str(j)] = ((reinf['Open'] - reinf['Close'].shift(j)) / reinf['Close'].shift(j)) * 100
            price_data[rt + ' + 1d ' + str(1)] = ((reinf['Open'] - reinf['Close']) / reinf['Open']) * 100
            price_data[rt + ' High ' + str(1)] = ((reinf['High'] - reinf['Low']) / reinf['Open']) * 100
            
            if len(np.where(price_data.columns == 'Adj Close')[0]) > 0 :
                price_data = price_data.drop(['Adj Close'], 1)
        
        price_data['change_in_price'] = (price_data['Close'] - price_data['Open'])
        
        price_data = price_data.reset_index(drop=False)
        price_data['Date'] = price_data['Date'].shift(-1)
        price_data['Date'].iloc[-1] = today.strftime('%Y - %m - %d')
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_data = price_data[price_data['Date'].dt.weekday < 5]
        price_data['Date'] = pd.to_datetime(price_data['Date']).dt.strftime('%Y-%m-%d')
        price_data.set_index('Date', inplace=True)
        
        self.price_data = price_data

        print(price_data.tail(1))
         
        return 
    
    def price_features(self):

        price_data = self.price_data

        # create ratios of downs last N days

        price_data['down'] = 0
        price_data.loc[price_data['change_in_price'] < 0, 'down'] = 1

        for k in range(1, 13):
            price_data['down_%s' % k] = price_data['down'].shift(k)

        price_data['percent_down_last_3_days'] = np.round(price_data[['down_1', 'down_2', 'down_3']].mean(axis=1), 2) * 100
        price_data['percent_down_last_6_days'] = np.round(price_data[['down_1', 'down_2', 'down_3', 'down_4', 'down_5',
                                                           'down_6']].mean(axis=1), 2) * 100
        price_data['percent_down_last_12_days'] = np.round(price_data[['down_1', 'down_2', 'down_3', 'down_4', 'down_5', 'down_6',
                           'down_7', 'down_8', 'down_9', 'down_10', 'down_11', 'down_12']].mean(axis=1), 2) * 100

        for k in range(1, 13):
            del price_data['down_%s' % k]

        per = [3, 5, 10, 20, 25, 50, 100, 200, 500]
        
        for n in per:
            mn = price_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = n).min())
            mx = price_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = n).max())
            
            price_data['max_diff'  + str(n)] = (price_data['Close'] - mx) / mx
            price_data['min_diff' + str(n)] = (price_data['Close'] - mn) / mn
            price_data['max_acc' + str(n)] = (price_data['max_diff' + str(n)] - price_data['max_diff' + str(n)].shift(1)) / price_data['Close']
            price_data['min_acc' + str(n)] = (price_data['min_diff' + str(n)] - price_data['min_diff' + str(n)].shift(1)) / price_data['Close']
            price_data['max_diff' + 'Fib1' + str(n)] = (mx - mn) / mn
            price_data['max_diff' + 'Fib61' + str(n)] = 0.6180 * (mx - mn) / mn
            price_data['max_diff' + 'Fib38' + str(n)] = 0.3820 * (mx - mn) / mn
            price_data['max_diff' + 'Fib12' + str(n)] = 0.1180 * (mx - mn) / mn
            price_data['max_diff' + 'Fib5' + str(n)] = 0.0486 * (mx - mn) / mn
            price_data['max_diff' + 'Fib2' + str(n)] = 0.0180 * (mx - mn) / mn
        
        #create 200 day rolling max / min
        mn = price_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = 200).min())
        mx = price_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = 200).max())
        price_data['max_diff' + str(200)] = (price_data['Close'] - mx) / mx
        price_data['min_diff' + str(200)] = (price_data['Close'] - mn) / mn
        price_data['max_acc' + str(200)] = (price_data['max_diff' + str(200)] - price_data['max_diff' + str(200)].shift(1)) / price_data['Close']
        price_data['min_acc' + str(200)] = (price_data['min_diff' + str(200)] - price_data['min_diff' + str(200)].shift(1)) / price_data['Close']
        
        #create 100 day rolling max / min
        mn = price_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window=100).min())
        mx = price_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window=100).max())
        price_data['max_diff' + str(100)] = (price_data['Close'] - mx) / mx
        price_data['min_diff' + str(100)] = (price_data['Close'] - mn) / mn
        price_data['max_acc' + str(100)] = (price_data['max_diff' + str(100)] - price_data['max_diff' + str(100)].shift(1)) / price_data['Close']
        price_data['min_acc' + str(100)] = (price_data['min_diff' + str(100)] - price_data['min_diff' + str(100)].shift(1)) / price_data['Close']
        
        #create 50 day rolling max / min
        mn = price_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window=50).min())
        mx = price_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window=50).max())
        price_data['max_diff' + str(50)] = (price_data['Close'] - mx) / mx
        price_data['min_diff' + str(50)] = (price_data['Close'] - mn) / mn
        price_data['max_acc' + str(50)] = (price_data['max_diff' + str(50)] - price_data['max_diff' + str(50)].shift(1)) / price_data['Close']
        price_data['min_acc' + str(50)] = (price_data['min_diff' + str(50)] - price_data['min_diff' + str(50)].shift(1)) / price_data['Close']
        
        #create 10 day rolling max / min
        mn = price_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = 10).min())
        mx = price_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = 10).max())
        price_data['max_diff' + str(10)] = (price_data['Close'] - mx) / mx
        price_data['min_diff' + str(10)] = (price_data['Close'] - mn) / mn
        price_data['max_acc' + str(10)] = (price_data['max_diff' + str(10)] - price_data['max_diff' + str(10)].shift(1)) / price_data['Close']
        price_data['min_acc' + str(10)] = (price_data['min_diff' + str(10)] - price_data['min_diff' + str(10)].shift(1)) / price_data['Close']
        
        #create 5 day rolling max / min
        mn = price_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = 5).min())
        mx = price_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = 5).max())
        price_data['max_diff' + str(5)] = (price_data['Close'] - mx) / mx
        price_data['min_diff' + str(5)] = (price_data['Close'] - mn) / mn
        price_data['max_acc' + str(5)] = (price_data['max_diff' + str(5)] - price_data['max_diff' + str(5)].shift(1)) / price_data['Close']
        price_data['min_acc' + str(5)] = (price_data['min_diff' + str(5)] - price_data['min_diff' + str(5)].shift(1)) / price_data['Close']
        
        #create 3 day rolling max / min
        mn = price_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = 3).min())
        mx  = price_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = 3).max())
        price_data['max_diff' + str(3)] = (price_data['Close'] - mx) / mx
        price_data['min_diff' + str(3)] = (price_data['Close'] - mn) / mn
        price_data['max_acc' + str(3)] = (price_data['max_diff' + str(3)] - price_data['max_diff' + str(3)].shift(1)) / price_data['Close']
        price_data['min_acc' + str(3)] = (price_data['min_diff' + str(3)] - price_data['min_diff' + str(3)].shift(1)) / price_data['Close']

        # Calculate the 2 day RSI
        n = 2
        
        # First make a copy of the data frame twice
        up_df, down_df = price_data[['symbol', 'change_in_price']].copy(), price_data[['symbol', 'change_in_price']].copy()
        
        # For up days, if the change is less than 0 set to 0.
        up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0
        
        # For down days, if the change is greater than 0 set to 0.
        down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0
        
        # We need change in price to be absolute.
        down_df['change_in_price'] = down_df['change_in_price'].abs()
        
        # Calculate the EWMA (Exponential Weighted Moving Average), meaning older values are given less weight compared to newer values.
        ewma_up = up_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
        ewma_down = down_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
        
        # Calculate the Relative Strength
        relative_strength = ewma_up / ewma_down
        
        # Calculate the Relative Strength Index
        relative_strength_index = 100.0 - (100.0 / (1.0  + relative_strength))
            
        # Add the info to the data frame.
        price_data['down_days2'] = down_df['change_in_price']
        price_data['up_days2'] = up_df['change_in_price']
        price_data['RSI2'] = relative_strength_index
        price_data['RSI2_signal'] = relative_strength_index - relative_strength_index.transform(lambda x: x.ewm(span = 7).mean())
        
        # Calculate the 3 day RSI
        n = 3
        
        # First make a copy of the data frame twice
        up_df, down_df = price_data[['symbol','change_in_price']].copy(), price_data[['symbol','change_in_price']].copy()
        
        # For up days, if the change is less than 0 set to 0.
        up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0
        
        # For down days, if the change is greater than 0 set to 0.
        down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0
        
        # We need change in price to be absolute.
        down_df['change_in_price'] = down_df['change_in_price'].abs()
        
        # Calculate the EWMA (Exponential Weighted Moving Average), meaning older values are given less weight compared to newer values.
        ewma_up = up_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
        ewma_down = down_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
        
        # Calculate the Relative Strength
        relative_strength = ewma_up / ewma_down
        
        # Calculate the Relative Strength Index
        relative_strength_index = 100.0 - (100.0 / (1.0  + relative_strength))
            
        # Add the info to the data frame.
        price_data['down_days3'] = down_df['change_in_price']
        price_data['up_days3'] = up_df['change_in_price']
        price_data['RSI3'] = relative_strength_index
        price_data['RSI3_signal'] = relative_strength_index - relative_strength_index.transform(lambda x: x.ewm(span = 7).mean())

        # Calculate the 7 day RSI
        n = 7
        
        # First make a copy of the data frame twice
        up_df, down_df = price_data[['symbol','change_in_price']].copy(), price_data[['symbol','change_in_price']].copy()
        
        # For up days, if the change is less than 0 set to 0.
        up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0
        
        # For down days, if the change is greater than 0 set to 0.
        down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0
        
        # We need change in price to be absolute.
        down_df['change_in_price'] = down_df['change_in_price'].abs()
        
        # Calculate the EWMA (Exponential Weighted Moving Average), meaning older values are given less weight compared to newer values.
        ewma_up = up_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
        ewma_down = down_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
        
        # Calculate the Relative Strength
        relative_strength = ewma_up  /  ewma_down
        
        # Calculate the Relative Strength Index
        relative_strength_index = 100.0 - (100.0  /  (1.0  + relative_strength))
        
        # Add the info to the data frame.
        price_data['down_days7'] = down_df['change_in_price']
        price_data['up_days7'] = up_df['change_in_price']
        price_data['RSI7'] = relative_strength_index
        price_data['RSI7_signal'] = relative_strength_index - relative_strength_index.transform(lambda x: x.ewm(span = 7).mean())
        
        # Calculate the 14 day RSI
        n = 14
        
        # First make a copy of the data frame twice
        up_df, down_df = price_data[['symbol','change_in_price']].copy(), price_data[['symbol','change_in_price']].copy()
        
        # For up days, if the change is less than 0 set to 0.
        up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0
        
        # For down days, if the change is greater than 0 set to 0.
        down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0
        
        # We need change in price to be absolute.
        down_df['change_in_price'] = down_df['change_in_price'].abs()
        
        # Calculate the EWMA (Exponential Weighted Moving Average), meaning older values are given less weight compared to newer values.
        ewma_up = up_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
        ewma_down = down_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
        
        # Calculate the Relative Strength
        relative_strength = ewma_up  /  ewma_down
        
        # Calculate the Relative Strength Index
        relative_strength_index = 100.0 - (100.0  /  (1.0  + relative_strength))
        
        # Add the info to the data frame.
        price_data['down_days14'] = down_df['change_in_price']
        price_data['up_days14'] = up_df['change_in_price']
        price_data['RSI14'] = relative_strength_index
        price_data['RSI14_signal'] = relative_strength_index - relative_strength_index.transform(lambda x: x.ewm(span = 7).mean())
        
        # Calculate the 30 day RSI
        n = 30
        
        # First make a copy of the data frame twice
        up_df, down_df = price_data[['symbol','change_in_price']].copy(), price_data[['symbol','change_in_price']].copy()
        
        # For up days, if the change is less than 0 set to 0.
        up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0
        
        # For down days, if the change is greater than 0 set to 0.
        down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0
        
        # We need change in price to be absolute.
        down_df['change_in_price'] = down_df['change_in_price'].abs()
        
        # Calculate the EWMA (Exponential Weighted Moving Average), meaning older values are given less weight compared to newer values.
        ewma_up = up_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
        ewma_down = down_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
        
        # Calculate the Relative Strength
        relative_strength = ewma_up  /  ewma_down
        
        # Calculate the Relative Strength Index
        relative_strength_index = 100.0 - (100.0  /  (1.0  + relative_strength))
        
        # Add the info to the data frame.
        price_data['down_days30'] = down_df['change_in_price']
        price_data['up_days30'] = up_df['change_in_price']
        price_data['RSI30'] = relative_strength_index
        price_data['RSI30_signal'] = relative_strength_index - relative_strength_index.transform(lambda x: x.ewm(span = 7).mean())
        
        # Calculate the Stochastic Oscillator
        n = 14
        
        # Make a copy of the High and Low column.
        low_14, high_14 = price_data[['symbol','Low']].copy(), price_data[['symbol','High']].copy()
        
        # Group by symbol, then apply the rolling function and grab the Min and Max.
        low_14 = low_14.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = n).min())
        high_14 = high_14.groupby('symbol')['High'].transform(lambda x: x.rolling(window = n).max())
        
        # Calculate the Stochastic Oscillator.
        k_percent = 100 * ((price_data['Close'] - low_14)  /  (high_14 - low_14))
        
        
        # Add the info to the data frame.
        price_data['low_14'] = low_14
        price_data['high_14'] = high_14
        price_data['k_percent'] = k_percent
        price_data['k_percent_signal'] = k_percent - k_percent.transform(lambda x: x.ewm(span = 7).mean())
        
        # Calculate the Williams %R
        n = 14
        
        # Make a copy of the High and Low column.
        low_14, high_14 = price_data[['symbol','Low']].copy(), price_data[['symbol','High']].copy()
        
        # Group by symbol, then apply the rolling function and grab the Min and Max.
        low_14 = low_14.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = n).min())
        high_14 = high_14.groupby('symbol')['High'].transform(lambda x: x.rolling(window = n).max())
        
        # Calculate William %R indicator.
        r_percent = ((high_14 - price_data['Close'])  /  (high_14 - low_14)) * -100
        
        # Add the info to the data frame.
        price_data['r_percent'] = r_percent
        
        # Calculate the MACD
        ema_26 = price_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span = 26).mean())
        ema_12 = price_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span = 12).mean())
        macd = ema_12 - ema_26
        
        # Calculate the EMA
        ema_9_macd = macd.ewm(span = 9).mean()
        
        # Store the data in the data frame.
        price_data['MACD'] = macd
        price_data['MACD_EMA'] = ema_9_macd
        price_data['MACD_signal'] = macd - ema_9_macd
        
        # Calculate the Price Rate of Change
        n = 5
        
        for i in range (n):
        # Calculate the Rate of Change in the Price, and store it in the Data Frame.
            price_data['Price_Rate_Of_Change' + str(i)] = price_data.groupby('symbol')['Close'].transform(lambda x: x.pct_change(periods = n))
        
        # Grab the Volume and Close column.
        Volume = price_data['Volume']
        change = price_data['Close'].diff()
        
        # intialize the previous OBV
        prev_obv = 0
        obv_values = []
        
        # calculate the On Balance Volume
        for i, j in zip(change, Volume):
            if i > 0:
                current_obv = prev_obv  + j
            elif i < 0:
                current_obv = prev_obv - j
            else:
                current_obv = prev_obv
        
        # OBV.append(current_OBV)
            prev_obv = current_obv
            obv_values.append(current_obv)
        
        price_data['On Balance Volume']  = obv_values
        
        # Calculating PPO 9
        ema_26 = price_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span = 26).mean())
        ema_12 = price_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span = 12).mean())
        PPO = ((ema_12 - ema_26) / ema_26) * 100
        
        #Calculating the PPO
        ema_9_PPO = PPO.ewm(span = 9).mean() 
        
        price_data['PPO'] = PPO
        price_data['PPO_EMA'] = ema_9_PPO
        price_data['PPO_signal'] = PPO - ema_9_PPO
        
        # Calculating PPO 5
        ema_26 = price_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span = 26).mean())
        ema_12 = price_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span = 12).mean())
        PPO = ((ema_12 - ema_26) / ema_26) * 100
        
        #Calculating the PPO
        ema_5_PPO = PPO.ewm(span = 5).mean() 
        
        price_data['PPO_5'] = PPO
        price_data['PPO_EMA_5'] = ema_5_PPO
        price_data['PPO_signal_5'] = PPO - ema_5_PPO
        
        # Calculating True Strength Index (TSI)
        
        #Double smoothed PC
        PCS = price_data.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = 25).mean())
        PCDS = PCS.transform(lambda x: x.ewm(span = 13).mean())
        
        #Double smoothed Absolute price change
        APC = price_data['change_in_price'].abs()
        FS = APC.transform(lambda x: x.ewm(span = 25).mean())
        SS = FS.transform(lambda x: x.ewm(span = 13).mean())
        
        # Calculation
        TSI = 100  *  (PCDS / SS)
        
        #Signal lines
        TSI_12_signal_line = TSI.transform(lambda x: x.ewm(span = 12).mean())
        TSI_7_signal_line = TSI.transform(lambda x: x.ewm(span = 7).mean())
        TSI_5_signal_line = TSI.transform(lambda x: x.ewm(span = 5).mean())
        TSI_3_signal_line = TSI.transform(lambda x: x.ewm(span = 3).mean())
        
        # RSI & Crossovers
        price_data['TSI'] = TSI
        price_data['TSI_12_signal'] = TSI - TSI_12_signal_line
        price_data['TSI_7_signal'] = TSI - TSI_7_signal_line
        price_data['TSI_3_signal'] = TSI - TSI_3_signal_line
        
        #Variations in RSI7
        n = 10
        
        for i in range(n):
        # Calculate the Rate of Change in the Price, and store it in the Data Frame.
            price_data['RSI3_Rate_Of_Change' + str(i)] = price_data.groupby('symbol')['RSI3'].transform(lambda x: x.pct_change(periods = n))
        
        #Variations in RSI7
        
        for i in range (n):
        # Calculate the Rate of Change in the Price, and store it in the Data Frame.
            price_data['RSI7_Rate_Of_Change' + str(i)] = price_data.groupby('symbol')['RSI7'].transform(lambda x: x.pct_change(periods = n))
        
        #Variations in RSI14
        
        for i in range (n):
        # Calculate the Rate of Change in the Price, and store it in the Data Frame.
            price_data['RSI14_Rate_Of_Change' + str(i)] = price_data.groupby('symbol')['RSI14'].transform(lambda x: x.pct_change(periods = n))
        
        #OBV rate of change
        n = 10
        
        for i in range(n):
        # Calculate the Rate of Change in the Price, and store it in the Data Frame.
            price_data['OBV_Rate_Of_Change' + str(i)] = price_data.groupby('symbol')['On Balance Volume'].transform(lambda x: x.pct_change(periods = n))
        
        #Bollinger bands 20
        Close = price_data[['symbol','Close']].copy()
        
        # Group by symbol, then apply the rolling function and grab the Min and Max.
        sma = Close.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 20).mean())
        
        price_data['sma'] = sma
        
        std = Close.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 20).std())
        upper_bb = sma + std * 2
        lower_bb = sma - std * 2
        
        price_data['upper_bb'], price_data['lower_bb'] = upper_bb, lower_bb
        price_data['bb_diff'] = upper_bb - lower_bb
        
        #Bollinger bands 10
        Close = price_data[['symbol', 'Close']].copy()
        
        # Group by symbol, then apply the rolling function and grab the Min and Max.
        sma = Close.groupby('symbol')['Close'].transform(lambda x: x.rolling(window=10).mean())
        
        price_data['sma'] = sma
        
        std = Close.groupby('symbol')['Close'].transform(lambda x: x.rolling(window=10).std())
        upper_bb = sma + std * 2
        lower_bb = sma - std * 2
        
        price_data['upper_bb' + str(10)], price_data['lower_bb' + str(10)] = upper_bb, lower_bb
        price_data['bb_diff' + str(10)] = upper_bb - lower_bb
        
        #Bollinger bands 5
        Close = price_data[['symbol', 'Close']].copy()
        
        # Group by symbol, then apply the rolling function and grab the Min and Max.
        sma = Close.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 5).mean())
        
        price_data['sma'] = sma
        
        std = Close.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 5).std())
        upper_bb = sma + std * 2
        lower_bb = sma - std * 2
        
        price_data['upper_bb' + str(5)], price_data['lower_bb' + str(5)] = upper_bb, lower_bb
        price_data['bb_diff' + str(5)] = upper_bb - lower_bb
        
        #Bollinger bands 50
        Close = price_data[['symbol','Close']].copy()
        
        # Group by symbol, then apply the rolling function and grab the Min and Max.
        sma = Close.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 50).mean())
        
        price_data['sma'] = sma
        
        std = Close.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 50).std())
        upper_bb = sma + std * 2
        lower_bb = sma - std * 2
        
        price_data['upper_bb' + str(50)], price_data['lower_bb' + str(50)] = upper_bb, lower_bb
        price_data['bb_diff' + str(50)] = upper_bb - lower_bb
         
        #DRX
        lookback = 14
        High = price_data['High']
        Low = price_data['Low']
        Close = price_data.groupby('symbol')['Close']
        
        plus_dm = price_data.groupby('symbol')['High'].diff()
        minus_dm = price_data.groupby('symbol')['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(High - Low)
        tr2 = pd.DataFrame(abs(High - Close.shift(1)))
        tr3 = pd.DataFrame(abs(Low - Close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.rolling(lookback).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha = 1 / lookback).mean() / atr)
        minus_di = abs(100  *  (minus_dm.ewm(alpha = 1 / lookback).mean() / atr))
        dx = (abs(plus_di - minus_di)  /  abs(plus_di  + minus_di)) * 100
        adx = ((dx.shift(1)  *  (lookback - 1)) + dx) / lookback
        adx_smooth = adx.ewm(alpha = 1 / lookback).mean()
        
        price_data['plus_di_14'] = plus_di
        price_data['minus_di_14'] = minus_di
        price_data['adx_14'] = adx_smooth
        price_data['adx_signal_14'] = adx_smooth - adx_smooth.transform(lambda x: x.ewm(span = 7).mean())
        
        lookback = 7
        High = price_data['High']
        Low = price_data['Low']
        Close = price_data.groupby('symbol')['Close']
        
        plus_dm = price_data.groupby('symbol')['High'].diff()
        minus_dm = price_data.groupby('symbol')['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(High - Low)
        tr2 = pd.DataFrame(abs(High - Close.shift(1)))
        tr3 = pd.DataFrame(abs(Low - Close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.rolling(lookback).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha = 1 / lookback).mean() / atr)
        minus_di = abs(100  *  (minus_dm.ewm(alpha = 1 / lookback).mean() / atr))
        dx = (abs(plus_di - minus_di)  /  abs(plus_di  + minus_di)) * 100
        adx = ((dx.shift(1)  *  (lookback - 1)) + dx) / lookback
        adx_smooth = adx.ewm(alpha = 1 / lookback).mean()
        
        price_data['plus_di_7'] = plus_di
        price_data['minus_di_7'] = minus_di
        price_data['adx_7'] = adx_smooth
        price_data['adx_signal_7'] = adx_smooth - adx_smooth.transform(lambda x: x.ewm(span = 4).mean())
        
        self.price_data = price_data
    
        return
        
    def signal_analysis(self) :
        
        price_data = self.price_data
        symbol = self.symbol
        
        APCA_API_KEY_ID = 'PKTNU9EVNS0IZ9ONX60K'
        APCA_API_SECRET_KEY = '3MUxZEaRMszDmRbcrzMUykffBkFV61R7QhvohdKN'
        APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
        APCA_API_DATA_URL = 'https://data.alpaca.markets'
        APCA_RETRY_MAX = 3	
        APCA_RETRY_WAIT = 3	
        APCA_RETRY_CODES = 429.504	
        api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY,APCA_API_BASE_URL)
        account = api.get_account()
        
        time = datetime.now().strftime("%m/%d/%Y")
        test = api.get_bars(symbol, TimeFrame.Hour , "2021-01-08", "2021-02-08", limit=10000, adjustment='raw').df
        
    def return_features (self) :
        
        price_data = self.price_data
        price_data = price_data.drop(['Open', 'Close', 'High', 'Low', 'Volume', 'symbol', 'change_in_price'], axis=1)
        price_data.to_csv('./data/%s_features_trading.csv' % self.symbol)
        
    def execute(self) :
        self.data_collect_price()
        self.price_features()
        self.return_features()
        
        return self.price_data


if __name__ == "__main__":
    main()
