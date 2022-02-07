#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 20:41:56 2021

@author: christian
"""

import time
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import glob
import os
import sys
import pandas as pd
import numpy as np
import warnings
import yfinance as yf
import shutil
from sklearn.metrics import accuracy_score, classification_report
import yahoo_fin
import yahoo_fin.stock_info as si
warnings.simplefilter(action = 'ignore')

def main():  
    today = datetime.today()
    traded = pd.read_csv('/Users/christian/Documents/Market_prediction/run/data/to_trade.csv')
    df_traded = pd.DataFrame()
    df_traded_sum = pd.DataFrame() 
    
    if len(traded):
        stocks = list(set(traded['Products'].tolist()))
        
        for stock in stocks :
            st = traded['Products'][traded['Products'] == stock].tolist()
            pred = traded['Side'][traded['Products'] == stock].tolist()
            prob = traded['Probability'][traded['Products'] == stock].tolist()
            prob_dist = traded['Prob_distance'][traded['Products'] == stock].tolist()
            
            start_date = datetime.today().date() - pd.Timedelta('5 days')
            end_date = datetime.today().date() + pd.Timedelta('1 days')
            st_data = get_price(stock, start_date)
            
            delta = (st_data['Close'] - st_data['Open'])/st_data['Open']
            delta = delta.iloc[-1]
            
            deltas = []
            outcomes = []
            
            for i in range(len(st)):
                deltas.append(delta)
                outcomes.append(np.sign(delta))
            
            df = pd.DataFrame({'Traded': st, 'predictions': pred,
                        'outcome': outcomes, 'Delta': deltas, 'Probability': prob, 
                        'Prob_distance': prob_dist}) 
            
            df_traded = df_traded.append(df, ignore_index=True)
            
            outcome = df_traded[df_traded['Traded'] == stock].sum().outcome
            outcome = 1*(outcome>1) + -1*(outcome<1)
            prediction = df_traded[df_traded['Traded'] == stock].sum().predictions
            prediction = 1*(prediction>1) + -1*(prediction<1)
            df = pd.DataFrame({'Traded': [stock], 'predictions': prediction,
                        'outcome': outcome}) 
            df_traded_sum = df_traded_sum.append(df, ignore_index=True) 
    
        print('\nAccuracy of sum of traded predictions :')
        print(np.round(accuracy_score(df_traded_sum['predictions'].tolist(), df_traded_sum['outcome'].tolist(), normalize = True) * 100,2))


    traded = pd.read_csv('/Users/christian/Documents/Market_prediction/run/data/trade_data.csv')
    traded['Side'] = traded['Probabilities']
    traded['Side'].loc[traded['Side'] > 0.5] = 1
    traded['Side'].loc[traded['Side'] < 0.5] = -1
    stocks = list(set(traded['Products'].tolist()))

    df_all = pd.DataFrame()
    df_all_sum = pd.DataFrame()
    for stock in stocks :
        st = traded['Products'][traded['Products'] == stock].tolist()
        pred = traded['Side'][traded['Products'] == stock].tolist()
        prob = traded['Probabilities'][traded['Products'] == stock].tolist()
        
        start_date = datetime.today().date() - pd.Timedelta('5 days')
        end_date = datetime.today().date() + pd.Timedelta('1 days')
        st_data = get_price(stock, start_date)
        
        delta = (st_data['Close'] - st_data['Open'])/st_data['Open']
        delta = delta.iloc[-1]
        
        deltas = []
        outcomes = []
        
        for i in range(len(st)):
            deltas.append(delta)
            outcomes.append(np.sign(delta))
        
        df = pd.DataFrame({'Traded': st, 'predictions': pred,
                    'outcome': outcomes, 'Delta': deltas, 'Probability': prob}) 
        
        df_all = df_all.append(df, ignore_index=True)
        
        outcome = df_all[df_all['Traded'] == stock].sum().outcome
        outcome = 1*(outcome>1) + -1*(outcome<1)
        prediction = df_all[df_all['Traded'] == stock].sum().predictions
        prediction = 1*(prediction>1) + -1*(prediction<1)
        df = pd.DataFrame({'Traded': [stock], 'predictions': prediction,
                    'outcome': outcome}) 
        df_all_sum = df_traded_sum.append(df, ignore_index=True) 

    print('\nAccuracy of all predictions :')
    print(np.round(accuracy_score(df_all['predictions'].tolist(), df_all['outcome'].tolist(), normalize = True) * 100,2))

    print('\nAccuracy of sum of all predictions :')
    print(np.round(accuracy_score(df_all_sum['predictions'].tolist(), df_all_sum['outcome'].tolist(), normalize = True) * 100,2))


def get_price(ticker, start_date):
    column_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    try :
        end_date = datetime.today().date() + pd.DateOffset(days=1)
        temp = yf.download(ticker, start=start_date, end=end_date, progress=False)
        temp = temp.reset_index()
        temp = temp.drop_duplicates(subset=['Date'])
        time.sleep(1)
    except :
        print('Using backup data API')
        start = start_date
        end = datetime.today().date() + pd.DateOffset(days=1)
        temp = si.get_data(ticker , start_date = start , end_date = end)
        temp = temp.reset_index()
        temp = temp.rename(columns={"index": "Date"})
        temp = temp.drop_duplicates(subset=['Date'])
        temp = temp.drop(['ticker'], axis=1)
        temp = temp.set_axis(column_names, axis=1, inplace=False)
        time.sleep(1)

    return temp



if __name__ == "__main__":
    main()
         
