#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 00:55:28 2021

@author: christian
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb 
import glob
import sys
import os, signal
import shutil
import random 
import time
from datetime import datetime, timedelta
import ast
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import warnings
import pytz
from email_updates_error import *
from email_updates_model_report import *
warnings.filterwarnings("ignore")

def main():
    os.system("python model_evaluate.py")
    nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
    date_sent = nyc_datetime - pd.Timedelta("1 days")
    for j in range(0,1000):
        maker = 0
        if j == 0 :
            maker = 1
            market(maker).stock_matrix()
            maker = 0
        
        # Send update email once a day
        nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
        start = nyc_datetime.replace(hour=16, minute=30, second=0,microsecond=0)
        end = nyc_datetime.replace(hour=17, minute=30, second=0,microsecond=0)
        if (nyc_datetime > start) & (nyc_datetime < end) & (nyc_datetime.date() > date_sent.date()) :
            print('\nSending email update on model quality\n')
            market(maker).emails()
            date_sent = nyc_datetime
            print('Completed\n')
        
        for k in range (0,1):
            market(maker).execute()
                
        os.system("python model_evaluate.py")
        
class market :
    def __init__(self, maker) :
        #self.verify_features_store()
        self.possibilities =  ['INTC', 'AMZN', 'FB', 'AAPL', 'DIS', 'TSLA', 'GOOG', 'GOOGL', 
                               'MSFT', 'NFLX', 'NVDA', 'BA', 'TWTR', 'AMD', 'WMT', 'JPM', 'SPY', 'QQQ', 'BAC', 'JNJ', 'PG', 'NKE' ]
        self.path = os.getcwd()
        self.n_iter = 5
        self.predict = random.sample(self.possibilities, 1)[0]
        self.use_n = random.randint(1, 10)
        self.price_data = pd.read_csv('./data/features_store.csv',',')
        self.all = glob.glob(os.getcwd() + '/models/*.{}'.format('csv'))
        self.flag = 0
        self.maker = maker
        
        rec_model = glob.glob('./data/record_model.{}'.format('csv'))
        if rec_model == [] :
            record = pd.DataFrame({'Date' : [], 'Model_name' : [], 
                            'Stock': [], 'Used' : [], 
                            'Parameters' : [], 'Long_accuracy' : [], 
                            'Long_AP_score' : [], 'Long_ROC_AUC': [], 'Long_Market_performance' : [], 
                            'Long_Model_performance' : [], 'Weekly_accuracy' : [], 
                            'Weekly_AP_score' : [], 'Weekly_ROC_AUC': [], 'Status' : []})
            record.to_csv('./data/record_model.csv', index = False)
            
        # Halt during pre-trading times
        nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
        start = nyc_datetime.replace(hour=7, minute=20, second=0,microsecond=0)
        end = nyc_datetime.replace(hour=9, minute=30, second=0,microsecond=0)
        if (nyc_datetime > start) & (nyc_datetime < end) :
            time.sleep(3600)
       
    def stock_matrix(self) :
        print('\nMaking stock links matrix\n')
        self.maker = 1
        self.use_weights = 0
        for stock_a in self.possibilities : 
            for stock_b in self.possibilities :
                print('For', stock_a, 'Using', stock_b)
                self.predict = stock_a
                self.use = [stock_a, stock_b]
                self.prep()
                self.make_model()
        os.system("python model_evaluate.py")
        if len(glob.glob('./data/stock_links.csv')) :
            os.system("rm ./data/stock_links.csv")
        
    def error_handling(self, error_message) :
        today = datetime.today()
        error_location = "model.py"
        error_report = pd.DataFrame({'Date' : today.strftime('%Y - %m - %d'), 'Code_name' : [error_location], 'Message' : [error_message]})
        error_report = error_report.set_index('Date')
        error(error_report)
        print(error_message)
        print('Code sleeping until user kill')
        time.sleep(10000000)
              
    def verify_features_store(self):
        price_data = pd.read_csv('./data/features_store.csv',',')
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        today = datetime.today()
        if price_data['Date'].iloc[0] < today - timedelta(days=50) :
            error_message = "The features store has not been updated for ten or more days, this is evaluated as a fatal error as it can lead to incorrect models, please update features store"
            self.error_handling(error_message)
            
    def use_stocks(self) :
        possibilities = self.possibilities

        if not len(glob.glob('./data/stock_links.csv')) :
            record = pd.read_csv('./data/record_model.csv')
            record = record.drop_duplicates(subset=['model_name'], keep='last')
            self.links = pd.DataFrame(columns = possibilities, index = possibilities)
            for stock_a in possibilities :
                line = 0
                for stock_b in possibilities :  
                    model = stock_a + '-' + str(100) + '-' + str(0) + '-' + stock_b + '-' + str(0)
                    if (record['trade_accuracy_test'][record['model_name'] == model].tolist() != []) :
                        value = record['Weekly_ROC_AUC'][record['Model_name'] == model].iloc[0]
                        #print(model, 'Has ROC AUC of', ROC)
                        self.links[stock_a].iloc[line] = value
                    line += 1
        
            self.links = self.links.reset_index()
            self.links.to_csv('./data/stock_links.csv', index = False)
            
        else :
            self.links = pd.read_csv('./data/stock_links.csv')

        self.links = self.links.sort_values(by=[self.predict], ascending=False)
        self.use = list(set(self.links['index'].head(self.use_n).tolist() + [self.predict]))
        print('\nPredicting : ', self.predict)
        print('Using the following stocks : ', self.use)
        
    def all_features (self) :
        price_data = self.price_data.drop(['Open','Close','Date','delta_class','delta','stock'],1)
        
        if self.maker == 0 :
            self.n_features = random.randint(30, 300)
        elif self.maker == 1 :
            self.n_features = 100
        
        return price_data.columns
    
    def prep(self) :
        price_data = self.price_data
        use = self.use 
        price_data['Date'] = pd.to_datetime(price_data['Date']) 
        dates = price_data['Date']
        dates = dates.drop_duplicates()
        last_date = dates.iloc[0]
        first_date = dates.iloc[200] 
        price_data = price_data.loc[(price_data['Date'] < last_date)]
        self.price_data_train = price_data[price_data['stock'].isin(use)].loc[(price_data['Date'] < first_date)]
        self.price_data_test = price_data[price_data['stock'] == self.predict].loc[(price_data['Date'] >= first_date)]
        self.y_train = self.price_data_train['delta_class'].tolist()
        self.y_test = self.price_data_test['delta_class'].tolist() 
        self.weights = []
                        
        if self.flag == 0 :
            self.features_name = self.all_features ()
            self.X_train = self.price_data_train[self.features_name]
            self.X_test = self.price_data_test[self.features_name]
        else :
            self.X_train = self.price_data_train[self.features_name]
            self.X_test = self.price_data_test[self.features_name]
            
        return 
        
    def make_model (self) :
        self.flag = 0
        y_train = self.y_train
        X_train = self.X_train     
        train_data=lgb.Dataset(X_train,label=y_train)
        
        self.train_names = '-'.join(self.use)
        self.model = self.predict + '-' + str(self.n_features) + '-' + str(self.use_weights) + '-' + self.train_names + '-' + str(0)
        leaves = [300,400, 500, 450, 550, 600, 650, 700]
        depth = [3,5,2,4,6]
        rate = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        bins = [10,20,30,50,100,200,300,400,500]
        param = {'num_leaves':400, 'objective':'binary','max_depth':2,'learning_rate':0.1,'max_bin':50, 'num_threads':12, 'verbose': -1, 'is_unbalance': True}
        num_round=50
        lgbm = lgb.train(param,train_data,num_round, verbose_eval=False)
        self.features_name = pd.Series(lgbm.feature_importance(), index=X_train.columns).sort_values(ascending=False).index.tolist()[:500]
        self.flag = 1
        
        self.prep()
        y_train = self.y_train
        X_train = self.X_train
        
        train_data = lgb.Dataset(X_train,label=y_train)
        param = {'num_leaves':random.choice(leaves), 'objective':'binary','max_depth':random.choice(depth),'learning_rate':random.choice(rate),'max_bin':random.choice(bins), 'num_threads':12, 'verbose': -1, 'is_unbalance': True}
        num_round = 50
        lgbm = lgb.train(param,train_data,num_round, verbose_eval=False)
        
        self.feature_imp = pd.Series(lgbm.feature_importance(), index=X_train.columns).sort_values(ascending=False).index.tolist()
        self.parameters = lgbm.params
        self.record()
        
    def record(self) :
        rec_features = glob.glob('./data/model_features.{}'.format('csv'))
        if rec_features == [] :
            record_features = pd.DataFrame()
            record_features.to_csv('./data/model_features.csv', index = False)
        else : 
            record_features = pd.read_csv('./data/model_features.csv')
        
        df = pd.DataFrame({self.model : self.feature_imp[:self.n_features]})
        record_features = pd.concat([record_features, df], axis=1)
        record_features.to_csv('./data/model_features.csv', index = False)
        
        record_features = pd.DataFrame({self.model : str(list(self.parameters.items()))}, index=[0])
        record_features.to_csv(os.getcwd() + '/models/' + str(self.model) + '.csv', index = False)
    
    def emails (self) :
        today = datetime.today() - pd.Timedelta("2 days")
        record = pd.read_csv('./data/record_model.csv').dropna()
        record = record.drop_duplicates(subset=['Model_name'], keep='first')
        record['Date'] = pd.to_datetime(record['Date'])
        record = record.loc[(record['Date'] > today)]
        
        Long_accuracy = np.array(record['Long_accuracy'].tolist())
        Long_performance = np.array(record['Long_Model_performance'].tolist())
        
        Weekly_ROC = np.array(record['Weekly_ROC_AUC'].tolist())
        Weekly_accuracy = np.array(record['Weekly_accuracy'].tolist())
        
        Trade_accuracy = np.array(record['Trade_accuracy'].tolist())
        Weekly_Trade_accuracy = np.array(record['Weekly_Trade_accuracy'].tolist())
        
        strong = len(np.where(Weekly_accuracy  > 60)[0])
        
        report = pd.DataFrame({'Date' : datetime.today().strftime('%Y - %m - %d'), 'Total_models' : [len(Long_accuracy)],'Strong_models' : [strong], 
                            'Accuracy_over_100_days': [np.mean(Long_accuracy)], 'Estimated performance_100_days' : [np.mean(Long_performance)], 
                            'Weekly_accuracy' : [np.mean(Weekly_accuracy)], 'Weekly_ROC' : [np.mean(Weekly_ROC)], 
                            'Weekly_traded_accuracy': [np.mean(Weekly_Trade_accuracy)]})
        
        report.to_csv('./data/model_report.csv', index = False)
        send_report()

                                 
    def execute(self) :
        start = datetime.now()
        self.use_weights = 0
        self.use_stocks()
        self.prep()
        self.make_model()
            
        stop = datetime.now()    
        print('Time for model creation : ', (stop - start))
        

if __name__ == "__main__":
    main()

