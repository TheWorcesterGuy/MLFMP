#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 20:41:56 2021

@author: christian
"""

import time
from datetime import datetime, timedelta
import glob
import os
import sys
import pandas as pd
import numpy as np
import warnings
import yfinance as yf
import shutil
from sklearn.metrics import accuracy_score, classification_report
warnings.simplefilter(action = 'ignore')
    
today = datetime.today()
traded = pd.read_csv('./data/to_trade.csv')
stocks = list(set(traded['Products'].tolist()))

df_traded = pd.DataFrame()
df_traded_sum = pd.DataFrame() 
for stock in stocks :
    st = traded['Products'][traded['Products'] == stock].tolist()
    pred = traded['Side'][traded['Products'] == stock].tolist()
    prob = traded['Probability'][traded['Products'] == stock].tolist()
    prob_dist = traded['Prob_distance'][traded['Products'] == stock].tolist()
    
    start_date = datetime.today().date() - pd.Timedelta('5 days')
    end_date = datetime.today().date() + pd.Timedelta('1 days')
    st_data = yf.download(stock, start = start_date, end = end_date, progress=False)
    
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

print('\nAccuracy of traded predictions :')
print(np.round(accuracy_score(df_traded['predictions'].tolist(), df_traded['outcome'].tolist(), normalize = True) * 100,2))


print('\nAccuracy of sum of traded predictions :')
print(np.round(accuracy_score(df_traded_sum['predictions'].tolist(), df_traded_sum['outcome'].tolist(), normalize = True) * 100,2))


traded = pd.read_csv('./data/trade_data.csv')
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
    st_data = yf.download(stock, start = start_date, end = end_date, progress=False)
    
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
         