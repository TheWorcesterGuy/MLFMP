#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 20:43:27 2021

@author: christian
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import sys
import time
import finnhub
import glob
import random
#warnings.simplefilter(action='ignore')


def main():
    print('\nDownloading market price data...\n')

    tickers = ['^FCHI','^STOXX50E','^AEX','000001.SS','^HSI','^N225','^BSESN','^SSMI','^IBEX',
               '^VVIX','^VIX','SPY', 'QQQ','GC=F','CL=F','SI=F','EURUSD=X','JPY=X', 'XLF','XLK','XLV','XLY',
               'INTC', 'AMZN', 'FB', 'AAPL', 'DIS', 'TSLA', 'GOOG', 
               'GOOGL', 'MSFT', 'NFLX', 'NVDA', 'BA', 'TWTR', 'AMD', 'WMT', 
               'JPM', 'SPY', 'QQQ', 'BAC', 'JNJ', 'PG', 'NKE']
    
    for ticker in tickers:
        nb_file = len(glob.glob('./data/TRADE_DATA/price_data/%s.csv' % ticker))
        if nb_file > 0:
            df = pd.read_csv('./data/TRADE_DATA/price_data/%s.csv' % ticker)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date')
            start_date = df['Date'].iloc[-2]  # to make sure we don't have any hole
        else:
            start_date = datetime(2014, 1, 1)
        
        price_data = get_price(ticker, start_date)
        
        if nb_file == 1:
            historical_data = pd.read_csv('./data/TRADE_DATA/price_data/%s.csv' % ticker)            
            price_data = historical_data.append([price_data])
            
        price_data['Date'] = pd.to_datetime(price_data['Date']).dt.strftime('%Y-%m-%d')
        price_data = price_data.drop_duplicates(subset=['Date'], keep='first')
        
        price_data.to_csv('./data/TRADE_DATA/price_data/%s.csv' % ticker, index=False)
    print('\nDone downloading market price data...\n')


def get_price(ticker, start_date):

    end_date = datetime.today().date() + pd.DateOffset(days=1)
    temp = yf.download(ticker, start=start_date, end=end_date, progress=False)
    temp = temp.reset_index()
    temp.drop_duplicates(subset=['Date'])
    time.sleep(1)

    return temp
    
if __name__ == "__main__":
    main()
    


