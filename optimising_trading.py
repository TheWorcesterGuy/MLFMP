#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 01:19:35 2021

@author: christian
"""

from datetime import datetime
from alpaca_trade_api.rest import TimeFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from scipy.interpolate import make_interp_spline
from sklearn import metrics
import time
from datetime import datetime, timedelta
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
import yfinance as yf
import re
import ast
warnings.simplefilter(action = 'ignore')
plt.rcParams["figure.figsize"] = 10, 5

money = 100000
all_p = pd.read_csv('./data/record_all_predictions.csv')
all_p['Date'] = pd.to_datetime(all_p['Date'])
print('\n\nAll accuracy is : %a perc'% np.round(accuracy_score(all_p['Prediction'], all_p['Outcome'], normalize = True) * 100,2))

traded = pd.read_csv('./data/record_traded.csv')
traded['Date'] = pd.to_datetime(traded['Date'])
print('\n\nAll accuracy is : %a perc'% np.round(accuracy_score(traded['Prediction'], traded['Outcome'], normalize = True) * 100,2))

df = all_p
df['Prob'] = df['Probability']
df['Prob'].mask(df['Prob'] < 0.5, 1-df['Prob'], inplace=True)

level = np.round(np.linspace(0.5,1,num=50),3)

acc = []
days = []
delta = []
for ll in level :
    df_ = df[df['Prob']>ll]
    df_ = df_.groupby(by=['Date','Traded']).mean()
    df_['Prediction'][df_['Prediction']>0] = 1
    df_['Prediction'][df_['Prediction']<0] = -1
    acc.append(np.round(accuracy_score(df_['Prediction'], df_['Outcome'], normalize = True) * 100,2))
    # print('\n\nAccuracy at %a is : %a perc'% (ll,acc[-1]))
    days.append(round(df_.groupby(by=['Date']).count().mean().Prediction,2))
    # print('Number of trades per day : %a' % days[-1])
    temp = df_['Prediction'] * df_['Delta']
    delta.append(np.round(temp.mean()*100,2))
    
    
plt.figure()
fig,ax = plt.subplots()
ax.plot(level, acc ,color="blue", label = 'accuracy')
plt.legend(loc='upper right')
ax.set_xlabel('Probability level',fontsize=14)
ax.set_ylabel('accuracy',color="blue",fontsize=14) 
ax2=ax.twinx()
ax2.plot(level, days,color="red", label = 'Trades per day')
ax2.set_ylabel('Trades per day',color="red",fontsize=14)
plt.legend(loc='upper left')
plt.show()


plt.figure()
fig,ax = plt.subplots()
ax.plot(level, acc ,color="blue", label = 'accuracy')
plt.legend(loc='upper right')
ax.set_xlabel('Probability level',fontsize=14)
ax.set_ylabel('accuracy',color="blue",fontsize=14) 
ax2=ax.twinx()
ax2.plot(level, delta,color="red", label = 'Return')
ax2.set_ylabel('Average daily delta',color="red",fontsize=14)
plt.legend(loc='upper left')
plt.show()



    