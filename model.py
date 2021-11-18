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
from features_report import *
import shap
import copy
warnings.filterwarnings("ignore")

def main():
    os.system("python model_evaluate.py")
    nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
    date_sent = nyc_datetime - pd.Timedelta("1 days")
    for j in range(0,1000):
        maker = 0
        if j == -10 :
            maker = 1
            market(maker).stock_matrix()
            maker = 0
        
        # Send update email once a day
        nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
        start = nyc_datetime.replace(hour=16, minute=30, second=0,microsecond=0)
        end = nyc_datetime.replace(hour=17, minute=30, second=0,microsecond=0)
        if (nyc_datetime > start) & (nyc_datetime < end) & (nyc_datetime.date() > date_sent.date()) :
            print('\nSending email update on model quality\n')
            features_report()
            top_50()
            market(maker).emails()
            date_sent = nyc_datetime
            print('Completed\n')
        
        for k in range (0,10):
            market(maker).execute()
                
        os.system("python model_evaluate.py")
        
class market :
    def __init__(self, maker) :
        self.verify_features_store()
        self.possibilities =  ['INTC', 'AMZN', 'FB', 'AAPL', 'DIS', 'TSLA', 'GOOG', 'GOOGL', 
                               'MSFT', 'NFLX', 'NVDA', 'BA', 'TWTR', 'AMD', 'WMT', 'JPM', 'SPY', 'QQQ', 'BAC', 'JNJ', 'PG', 'NKE' ]
        self.path = os.getcwd()
        self.n_iter = 5
        self.predict = random.sample(self.possibilities, 1)[0]
        self.use_n = random.randint(1, 10)
        self.price_data = pd.read_csv('./data/features_store.csv',',')
        self.price_data = self.price_data.dropna(axis=1, thresh=int(np.shape(self.price_data)[0]*0.9)).dropna()
        self.all = glob.glob(os.getcwd() + '/models/*.{}'.format('csv'))
        self.flag = 0
        self.maker = maker
        
        #Halt during pre-trading times
        nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
        start = nyc_datetime.replace(hour=7, minute=20, second=0,microsecond=0)
        end = nyc_datetime.replace(hour=9, minute=30, second=0,microsecond=0)
        if (nyc_datetime > start) & (nyc_datetime < end) :
            os.system('rm ./data/stock_links.csv')
            time.sleep(3600)
       
    def stock_matrix(self) :
        print('\nMaking stock links matrix\n')
        self.maker = 1
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
        if price_data['Date'].iloc[0] < today - timedelta(days=5) :
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
                    model = stock_a + '-' + stock_a + '-' + stock_b + '-' + str(0)
                    if (record['trade_accuracy_test'][record['model_name'] == model].tolist() != []) :
                        ROC_test = record['ROC_test'][record['model_name'] == model].iloc[0]
                        accuracy_test = record['accuracy_test'][record['model_name'] == model].iloc[0]
                        #print(model, 'Has ROC AUC of', ROC)
                        self.links[stock_a].iloc[line] = (ROC_test + accuracy_test)/2
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
                        
        self.features_name = self.all_features ()
        self.X_train = self.price_data_train[self.features_name]
        self.X_test = self.price_data_test[self.features_name]
            
        return 
        
    def make_model (self) :
        
        self.train_names = '-'.join(self.use)
        self.model = self.predict + '-' + self.train_names + '-' + str(0)
        leaves = [450]
        depth = [6]
        rate = [0.01]
        bins = [200]
        
        self.prep()
        y_train = self.y_train
        X_train = self.X_train
        
        train_data = lgb.Dataset(X_train,label=y_train)
        param = {'num_leaves':random.choice(leaves), 'objective':'binary','max_depth':random.choice(depth),
                 'learning_rate':random.choice(rate),'max_bin':random.choice(bins),
                 'num_threads':10, 'verbose': -1, 'is_unbalance': True}
        num_round = 100
        lgbm = lgb.train(param,train_data,num_round, verbose_eval=False)
        
        explainer = shap.TreeExplainer(lgbm)
        shap_values = explainer.shap_values(self.X_test)
        values = np.sum(np.abs(shap_values), axis=0)
        values = np.mean(values, axis=0)
        df = pd.DataFrame({'features' : self.X_test.columns, 'value' : values}).sort_values(by=['value'], ascending=False)
        self.feature_imp = df[df['value'] != 0]
        
        self.parameters = lgbm.params
        self.record()
        #If plotting wanted
        # shap.summary_plot(shap_values,self.X_test,max_display=35)
        # explainer = shap.TreeExplainer(lgbm)
        # shap_values1 = explainer(self.X_test)
        # shap_values2 = copy.deepcopy(shap_values1)
        # shap_values2.values = shap_values2.values[:,:,1]
        # shap_values2.base_values = shap_values2.base_values[:,1]
        # shap.plots.beeswarm(shap_values2, max_display=35)
        
    def record(self) :
        rec_features = glob.glob('./data/model_features.{}'.format('csv'))
        if rec_features == [] :
            record_features = pd.DataFrame()
            record_features.to_csv('./data/model_features.csv', index = False)
        else : 
            record_features = pd.read_csv('./data/model_features.csv')
        
        df = pd.DataFrame({self.model : self.feature_imp['features'].tolist()})
        record_features = pd.concat([record_features, df], axis=1)
        record_features.to_csv('./data/model_features.csv', index = False)
        
        record_features = pd.DataFrame({self.model : str(list(self.parameters.items()))}, index=[0])
        record_features.to_csv(os.getcwd() + '/models/' + str(self.model) + '.csv', index = False)
    
    def emails (self) :
        today = datetime.today() - pd.Timedelta("2 days")
        record = pd.read_csv('./data/record_model.csv').dropna()
        record = record.drop_duplicates(subset=['model_name'], keep='last')
        record['date'] = pd.to_datetime(record['date'])
        record = record.loc[(record['date'] > today)]
        
        accuracy_test = np.array(record['accuracy_test'].tolist())
        ROC_test = np.array(record['ROC_test'].tolist())
        trade_accuracy_test = np.array(record['trade_accuracy_test'].tolist())
        model_performance_test = np.array(record['model_performance_test'].tolist())
        
        accuracy_live = np.array(record['accuracy_live'].tolist())
        ROC_live = np.array(record['ROC_live'].tolist())
        trade_accuracy_live = np.array(record['trade_accuracy_live'].tolist())
        model_performance_live = np.array(record['model_performance_live'].tolist())
        
        report = pd.DataFrame({'date':[datetime.today().strftime('%Y - %m - %d')],
                            'accuracy_test' : [np.round(np.mean(accuracy_test),2)], 'ROC_test': [np.round(np.mean(ROC_test),2)], 
                            'trade_accuracy_test' : [np.round(np.mean(trade_accuracy_test),2)], 'model_performance_test' : [np.round(np.mean(model_performance_test),2)], 
                            'accuracy_live' : [np.round(np.mean(accuracy_live),2)], 'ROC_live': [np.round(np.mean(ROC_live),2)], 
                            'trade_accuracy_live' : [np.round(np.mean(trade_accuracy_live),2)], 'model_performance_live' : [np.round(np.mean(model_performance_live),2)]})
        
        report.to_csv('./data/model_report.csv', index = False)
        send_report()

                                 
    def execute(self) :
        start = datetime.now()
        self.use_stocks()
        self.prep()
        self.make_model()
            
        stop = datetime.now()    
        print('Time for model creation : ', (stop - start))
        

if __name__ == "__main__":
    main()

