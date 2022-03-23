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
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
warnings.simplefilter(action = 'ignore')
plt.rcParams["figure.figsize"] = 10, 5

def main() :
    f_acc(p = 0.6, l= 60, probability = True, acc_level = False, plots = True, verbose = True)

    
def f_acc(p = 0.6, l = 60, probability = True, acc_level = False, plots = False, verbose = False):
    account = pd.read_csv('./data/account.csv')
    account['Date'] = pd.to_datetime(account['Date'])
    avoid = account[account['Trade_factor'] != 1]['Date']
    
    all_p = pd.read_csv('./data/record_all_predictions.csv')
    all_p['Date'] = pd.to_datetime(all_p['Date'])
    today = datetime.now()
    start = datetime(2022, 1, 9, 0, 0, 0, 0)
    all_p = all_p[all_p['Date'] > start]
    all_p = all_p[~all_p['Date'].isin(avoid)]
    
    # today = datetime.now()
    # recent = today - timedelta(days=1)
    # all_p = all_p[all_p['Date'] <= recent].dropna()
    
    if verbose :
        print('\n\nAll accuracy is : %a perc'% np.round(accuracy_score(all_p['Prediction'], all_p['Outcome'], normalize = True) * 100,2))
    
    df = all_p
    df['Prob'] = df['Probability']
    df['Prob'].mask(df['Prob'] < 0.5, 1-df['Prob'], inplace=True)
    
    level =  np.round(np.linspace(0.5,1,num=1000),3)
    
    acc = []
    days = []
    delta = []
    for ll in level :
        df_ = df[df['Prob']>ll]
        df_ = df_.groupby(by=['Date','Traded']).mean()
        df_['Prediction'][df_['Prediction']>0] = 1
        df_['Prediction'][df_['Prediction']<=0] = -1
        acc.append(np.round(accuracy_score(df_['Prediction'], df_['Outcome'], normalize = True) * 100,2))
        # print('\n\nAccuracy at %a is : %a perc'% (ll,acc[-1]))
        days.append(round(df_.groupby(by=['Date']).count().mean().Prediction,2))
        # print('Number of trades per day : %a' % days[-1])
        temp = df_['Prediction'] * df_['Delta']
        delta.append(np.round(temp.mean()*100,2))
        if np.isnan(days[-1]) :
            break
    
    def Exp(x, A, B, C):
        y = C + A*np.exp(B*x)
        return y
    
    def squ(x, A, B, C, D,E):
        y = A + B*x + C*x**2 + D*x**3 + E*x**4
        return y
    
    acc = acc[:-1]
    level = level[:len(acc)]
    days = days[:len(acc)]
    delta = delta[:len(acc)]
    acc = np.maximum.accumulate(acc)
    delta = np.maximum.accumulate(delta)
    true_level = level
    true_acc = acc
    df = pd.DataFrame({'level': level,'acc' : np.maximum.accumulate(acc), 'days':days, 'delta': delta})
    
    try :
        parameters, covariance = curve_fit(Exp, level, acc)
        level = np.linspace(0.5,max(level),1000)
        acc = parameters[2]+parameters[0]*np.exp(parameters[1]*level)   
    except :
        print('\nExponential fit failed, trying interpolation\n')
        inter = interp1d(level, acc, fill_value="extrapolate",kind='linear')
        level = np.linspace(0.5,max(level),1000)
        acc = inter(level)
    
    if plots :
        plt.figure()
        plt.plot(level, acc, 'o', label = "Accuracy from regression")
        plt.plot(true_level, true_acc, '*', label = "True Accuracy")
        plt.xlabel('Probability level')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left')
        
        plt.figure()
        fig,ax = plt.subplots()
        ax.plot(level, acc ,color="blue", label = 'accuracy')
        plt.legend(loc='upper right')
        ax.set_xlabel('Probability level',fontsize=14)
        ax.set_ylabel('accuracy',color="blue",fontsize=14) 
        ax2=ax.twinx()
        ax2.plot(true_level, days,color="red", label = 'Trades per day')
        ax2.set_ylabel('Trades per day',color="red",fontsize=14)
        plt.legend(loc='upper left')
        #plt.savefig('./Images/trades per day per level.png')
        plt.show()
        
        
        plt.figure()
        fig,ax = plt.subplots()
        ax.plot(level, acc ,color="blue", label = 'accuracy')
        plt.legend(loc='upper right')
        ax.set_xlabel('Probability level',fontsize=14)
        ax.set_ylabel('accuracy',color="blue",fontsize=14) 
        ax2=ax.twinx()
        ax2.plot(true_level, delta ,color="red", label = 'Return')
        ax2.set_ylabel('Average daily delta',color="red",fontsize=14)
        plt.legend(loc='upper left')
        #plt.savefig('./Images/average return per level.png')
        plt.show()
    
    try :
        p_level = level[np.where(np.array(acc)>l)[0][0]]
        d_level = days[np.where(np.array(true_acc)>l)[0][0]]
    except :
        p_level = 0.64
        print('\nFailed to find level, data incompatable setting probability level to {}'.format(p_level))
        d_level = False
        
    if p < 0.5 :
        p = 1-p
        
    try :
        acc_p = acc[np.where(np.array(level)>p)[0][0]]
    except :
        print('No accuracy level found, setting level to 60%')
        acc_p=60
    
    if verbose :
        print('\nUsing probability threshold of %a\n' %np.round(p_level,2))
        print('\nAverage number of trades per day at this level %a\n' %d_level)
        print('\nAccuracy at probability level %a\n' %np.round(acc_p,2))
    
    if p_level < 0.55 :
        print('\nFound probability level to low ({}), raising it to 0.64'.format(p_level))
        p_level = 0.64
        
    if p_level > 0.7 :
        print('\nFound probability level to high ({}), lowering it to 0.64'.format(p_level))
        p_level = 0.64
    
    if probability :
        return np.round(acc_p/50,3)
    
    if acc_level :
        return np.round(p_level,3)

if __name__ == "__main__":
    main()



    