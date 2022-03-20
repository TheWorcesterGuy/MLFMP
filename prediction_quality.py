from calendar import month
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
import lightgbm as lgb 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns

warnings.simplefilter(action='ignore')

def main():
    pred_quality()
    
def pred_quality():
    possibilities =  ['INTC', 'AMZN', 'FB', 'AAPL', 'DIS', 'TSLA', 'GOOG', 'GOOGL', 
                                'MSFT', 'NFLX', 'NVDA', 'TWTR', 'AMD', 'WMT', 'JPM', 'SPY', 'QQQ', 'BAC', 'PG']


    preds = pd.DataFrame()
    df = pd.read_csv('./data/record_all_predictions.csv')
    df = df.groupby(['Date','Traded']).mean()['Probability'].reset_index()
    df = df.set_index(['Date'])
    size = int(len(df)*0.008)

    temp = pd.DataFrame()
    for p in possibilities :
        preds = pd.concat([preds, df[df['Traded']==p]['Probability']], axis=1) 
        preds.columns = [*preds.columns[:-1], p]
        #preds[p] = df[df['Traded']==p]['Probability']
    
        
    mon = pd.read_csv('./data/account.csv')
    mon['Change_account_%'][mon['Change_account_%']>0] = 1
    mon['Change_account_%'][mon['Change_account_%']<0] = -1
    mon = mon.set_index(['Date'])
    
    preds['mean']  = preds.mean(axis = 1)
    preds['std'] = preds.std(axis = 1)
    preds['target'] = mon['Change_account_%']
      
    preds = preds.dropna(subset=['target'])
    preds = preds.sample(frac=1)

    X_train = preds.drop(columns=['target']).iloc[:-size]
    y_train = preds['target'].iloc[:-size]

    X_test = preds.drop(columns=['target']).iloc[-size:]
    y_test = preds['target'].iloc[-size:]
    print(X_test)

    # X_train = preds.drop(columns=['target']).iloc[size:]
    # y_train = preds['target'].iloc[size:]

    # X_test = preds.drop(columns=['target']).iloc[:size]
    # y_test = preds['target'].iloc[:size]


    train_data = lgb.Dataset(X_train,label=y_train)
    param = {'objective':'binary','verbose': -1}

    num_round = 100
    lgbm = lgb.train(param,train_data,num_round, verbose_eval=False)

    y_probs = lgbm.predict(X_test)

    y_pred = []
    for p in y_probs:
        if p > .5 :
            y_pred.append(1)
        else :  
            y_pred.append(-1)

    #print(y_probs)
    #print(y_test)

    score = accuracy_score(y_test, y_pred, normalize = True)
    print('Accuracy is ', score, ' for {} days'.format(len(y_pred)))

    def plotImp(model, X , num = 20, fig_size = (40, 20)):
        feature_imp = pd.DataFrame({'Value':model.feature_importance(),'Feature':X.columns})
        plt.figure(figsize=fig_size)
        sns.set(font_scale = 5)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                            ascending=False)[0:num])
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('lgbm_importances-01.png')
        plt.show()
        #plt.close('all')
        
    X_train = preds.drop(columns=['target'])
    y_train = preds['target']
    num_round = 100
    train_data = lgb.Dataset(X_train,label=y_train)
    lgbm = lgb.train(param,train_data,num_round, verbose_eval=False)
    
    df = pd.read_csv('./data/trade_data.csv')
    df = df.groupby(['Date','Products']).mean()['Probabilities'].reset_index()
    df = df.set_index(['Date'])
    preds = pd.DataFrame()
    for p in possibilities :
        preds[p] = df[df['Products']==p]['Probabilities']

    preds['mean']  = preds.mean(axis = 1)
    preds['std'] = preds.std(axis = 1)
    #print(preds)
    quality = lgbm.predict(preds)
    print('Todays quality is ', np.round(quality[0],2))
    
    #plotImp(lgbm, X_test , num = 20, fig_size = (40, 20))
    
    return quality

if __name__ == "__main__":
    main()