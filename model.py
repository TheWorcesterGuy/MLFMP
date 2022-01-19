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
import re
import MacTmp
from itertools import product
warnings.filterwarnings("ignore")

def main():
    #os.system("python3 update_features_store.py")
    #market().stock_matrix()
    os.system("python3 model_evaluate.py")
    nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
    date_sent = nyc_datetime - pd.Timedelta("1 days")
    
    for j in range(0,1000):   
        
        # Send update email once a day
        nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
        start = nyc_datetime.replace(hour=16, minute=0, second=0,microsecond=0)
        end = nyc_datetime.replace(hour=17, minute=0, second=0,microsecond=0)
        if (nyc_datetime > start) & (nyc_datetime < end) & (nyc_datetime.date() > date_sent.date()) :
            print('\nSending email update on model quality\n')
            os.system('rm ./data/stock_links.csv')
            features_report()
            top_50()
            market().emails()
            date_sent = datetime.now(pytz.timezone('US/Eastern'))
        
        # Create stock links matrix on the Weekend
        nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
        start = nyc_datetime.replace(hour=1, minute=0, second=1,microsecond=0)
        end = nyc_datetime.replace(hour=1, minute=30, second=0,microsecond=0)
        if (nyc_datetime > start) & (nyc_datetime < end) & (nyc_datetime.weekday() == 6):
            market().stock_matrix()
        
        market().execute()
                
        os.system("python model_evaluate.py")
        
class market :
    def __init__(self) :
        self.possibilities =  ['INTC', 'AMZN', 'FB', 'AAPL', 'DIS', 'TSLA', 'GOOG', 'GOOGL', 
                               'MSFT', 'NFLX', 'NVDA', 'TWTR', 'AMD', 'WMT', 'JPM', 'SPY', 'QQQ', 'BAC', 'PG']
        self.price_data = pd.read_csv('./data/features_store.csv',',')
        self.price_data = self.price_data.dropna(axis=1, thresh=int(np.shape(self.price_data)[0]*0.95))
        self.all = glob.glob(os.getcwd() + '/models/*.{}'.format('csv'))
        self.CPU_high_counter = 0
        self.threads = 10
       
    def stock_matrix(self) :
        self.mode = 10
        print('\nMaking stock links matrix\n')
        for stock_a in self.possibilities : 
            for stock_b in self.possibilities :
                print('For', stock_a, 'Using', stock_b)
                self.predict = stock_a
                self.use = [stock_a, stock_b]
                self.use = list(set(self.use))
                self.prep()
                self.make_model()
        os.system("python model_evaluate.py")
        if len(glob.glob('./data/stock_links.csv')) :
            os.system("rm ./data/stock_links.csv")
            
    def use_stocks(self) :
        possibilities = self.possibilities

        if not len(glob.glob('./data/stock_links.csv')) :
            record = pd.read_csv('./data/record_model.csv')
            record = record.drop_duplicates(subset=['model_name'], keep='last')
            record['model_name'] = record['model_name'].str.replace('-0', '').str.replace('.', '').str.replace('\d+', '')
            self.links = pd.DataFrame(columns = possibilities, index = possibilities)
            for stock_a in possibilities :
                line = 0
                for stock_b in possibilities :  
                    model = stock_a + '-' + stock_a + '-' + stock_b
                    if (record['trade_accuracy_test'][record['model_name'] == model].tolist() != []) :
                        ROC_test = record['ROC_test'][record['model_name'] == model].iloc[0]
                        accuracy_test = record['accuracy_test'][record['model_name'] == model].iloc[0]
                        self.links[stock_a].iloc[line] = (ROC_test + accuracy_test)/2
                        
                    model = stock_a + '-' + stock_b + '-' + stock_a
                    if (record['trade_accuracy_test'][record['model_name'] == model].tolist() != []) :
                        ROC_test = record['ROC_test'][record['model_name'] == model].iloc[0]
                        accuracy_test = record['accuracy_test'][record['model_name'] == model].iloc[0]
                        self.links[stock_a].iloc[line] = (ROC_test + accuracy_test)/2
                        
                    model = stock_a + '-' + stock_b
                    if (record['trade_accuracy_test'][record['model_name'] == model].tolist() != []) :
                        ROC_test = record['ROC_test'][record['model_name'] == model].iloc[0]
                        accuracy_test = record['accuracy_test'][record['model_name'] == model].iloc[0]
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
        
    def features (self) :
        price_data = self.price_data.drop(['Date','delta_class','delta','stock'],1)
                
        return price_data.columns
        
    def prep(self) :
        price_data = self.price_data
        use = self.use 
        price_data['Date'] = pd.to_datetime(price_data['Date']) 
        dates = price_data['Date']
        dates = dates.drop_duplicates()
        last_date = dates.iloc[0]
        first_date = dates.iloc[10] 
        price_data = price_data.loc[(price_data['Date'] < last_date)]
        self.price_data_train = price_data[price_data['stock'].isin(use)].loc[(price_data['Date'] < first_date)]
        self.price_data_test = price_data[price_data['stock'] == self.predict].loc[(price_data['Date'] >= first_date)]
        self.y_train = self.price_data_train['delta_class'].tolist()
        self.y_test = self.price_data_test['delta_class'].tolist()                 
        self.X_train = self.price_data_train[self.features()]
        self.X_test = self.price_data_test[self.features()]
            
        return 
        
    def make_model (self) :
        ml_parameters = pd.read_csv('./data/parameters.csv').iloc[0]        
        if self.mode == 1 :
            shaps = np.round(np.logspace(-3.5,-1,num=20),4)
            leaves = np.round(np.linspace(5,1000,30),0)
            depth = np.round(np.linspace(2,10,10),0)       
            rate = np.round(np.logspace(-2.5,-0.5,num=20),4)         
            bins = np.round(np.linspace(100,900,30),0)   
            is_unbalance = [0, 1]
            
            choice = random.choice(list(product(shaps,leaves,depth,rate,bins,is_unbalance)))
            
            self.shaps = choice[0]
            leaves = int(choice[1])
            depth = int(choice[2])      
            rate = choice[3]         
            bins = int(choice[4])  
            is_unbalance = bool(choice[5])
        else :
            self.shaps = ml_parameters.shaps
            leaves = int(ml_parameters.num_leaves)
            depth = int(ml_parameters.max_depth)      
            rate = ml_parameters.learning_rate    
            bins = int(ml_parameters.bins)  
            is_unbalance = bool(ml_parameters.is_unbalance)
            
        self.train_names = '-'.join(self.use)
        self.model = self.predict + '-' + self.train_names + '-' + str(self.shaps)
        
        self.prep()
        y_train = self.y_train
        X_train = self.X_train
        
        train_data = lgb.Dataset(X_train,label=y_train)
        param = {'num_leaves':leaves, 'objective':'binary','max_depth':depth,
                 'learning_rate':rate,'max_bin':bins,
                 'num_threads':self.threads, 'verbose': -1, 'is_unbalance': is_unbalance}
        
        num_round = 100
        lgbm = lgb.train(param,train_data,num_round, verbose_eval=False)
        
        explainer = shap.TreeExplainer(lgbm)
        shap_values = explainer.shap_values(self.X_test)
        values = np.sum(np.abs(shap_values), axis=0)
        values = np.mean(values, axis=0)
        df = pd.DataFrame({'features' : self.X_test.columns, 'value' : values}).sort_values(by=['value'], ascending=False)
        self.feature_imp = df[df['value']>self.shaps]
        print('Shaps level : %a' %self.shaps)
        print(lgbm.params)
        if np.shape(self.feature_imp)[0]>0:
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
        # plt.show()
        
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

        today = datetime.today() - pd.Timedelta("2 days")
        record = pd.read_csv('./data/record_model.csv').dropna()
        record['date'] = pd.to_datetime(record['date'])
        record = record.loc[(record['date'] > today)]
        selected_models = np.round(100 * record['status'].sum()/record['status'].count(),2)
        
        report = pd.DataFrame({'date':[datetime.today().strftime('%Y - %m - %d')],
                            'accuracy_test' : [np.round(np.mean(accuracy_test),2)], 'ROC_test': [np.round(np.mean(ROC_test),2)], 
                            'trade_accuracy_test' : [np.round(np.mean(trade_accuracy_test),2)], 'model_performance_test' : [np.round(np.mean(model_performance_test),2)], 
                            'accuracy_live' : [np.round(np.mean(accuracy_live),2)], 'ROC_live': [np.round(np.mean(ROC_live),2)], 
                            'trade_accuracy_live' : [np.round(np.mean(trade_accuracy_live),2)], 'model_performance_live' : [np.round(np.mean(model_performance_live),2)],
                            'selected_models_%' : [selected_models]})
        
        report.to_csv('./data/model_report.csv', index = False)
        send_report()

    def selector(self):
        def length(x):
            return len(eval(x))
        
        try :
            record = pd.read_csv('./data/record_model.csv')
            record['date'] = pd.to_datetime(record['date'])
            record = record.drop_duplicates(subset=['model_name'], keep='last')
            record['length'] = record['used'].apply(length)
            record_ = record[record['length']==len(self.use)]
            record__ = record_[record_['stock'] == self.predict]
            if len(record__)>10 :
                record = record__
            elif len(record_)>10 :
                record = record_
                
            # today = datetime.now()
            # recent = today - timedelta(days=5)
            # record = record[record['date'] > recent].dropna()
                
            # percentage_days = 10
            # record = record[record['days_traded_test'] > int(150*percentage_days/100)]
            # record = record[record['days_traded_live'] > int(100*percentage_days/100)]
            
            models = record['model_name'].tolist()
            accuracy_test = np.array(record['trade_accuracy_test'].to_list())
            accuracy_live = np.array(record['trade_accuracy_live'].to_list())
            ROC_live = np.array(record['ROC_live'].to_list())
            ROC_test = np.array(record['ROC_test'].to_list())
            metric = list(record['trade_accuracy_test'] * np.log10(record['days_traded_test']))
            
            parameters = record.parameters.to_list()
            shaps = [float(model.split('-')[-1]) for model in models]
            num_leaves = [int(eval(x)[0][1]) for x in parameters]
            max_depth = [int(eval(x)[2][1]) for x in parameters]
            learning_rate = [eval(x)[3][1] for x in parameters]
            bins = [int(eval(x)[4][1]) for x in parameters]
            is_unbalance = [int(eval(x)[7][1]) for x in parameters]
            
            df = pd.DataFrame({'shaps' : shaps, 'num_leaves' : num_leaves, 'max_depth': max_depth, 
                               'learning_rate' :learning_rate, 'bins': bins,
                               'is_unbalance': is_unbalance, 'metric': metric})
            df = df[df['shaps']>0]
            best = df.groupby(['shaps','num_leaves', 'max_depth','learning_rate','bins','is_unbalance']).mean().sort_values(['metric'],ascending=False).index[0]
                
            ml_parameters = pd.DataFrame()
            ml_parameters['shaps'] = [best[0]]
            ml_parameters['num_leaves'] = [best[1]]
            ml_parameters['max_depth'] = [best[2]]
            ml_parameters['learning_rate'] = [best[3]]
            ml_parameters['bins'] = best[4]
            ml_parameters['is_unbalance'] = [best[5]]
                
        except:
            print('\nFailed to extract parameters from model record - Using preset parameters\n')
            ml_parameters = pd.DataFrame()
            ml_parameters['shaps'] = [0.02]
            ml_parameters['num_leaves'] = [400]
            ml_parameters['max_depth'] = [2]
            ml_parameters['learning_rate'] = [0.01]
            ml_parameters['bins'] = [50]
            ml_parameters['is_unbalance'] = [1]
                
        ml_parameters.to_csv('./data/parameters.csv', index=False)
        
    def execute(self) :
        time_m = 0
        number_of_models = 50
        for k in range (0,number_of_models):
            
            nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
            if (nyc_datetime.weekday() not in [5,6]) :
                #Halt during pre-trading times
                nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
                start = nyc_datetime.replace(hour=7, minute=30, second=0,microsecond=0)
                end = nyc_datetime.replace(hour=9, minute=30, second=0,microsecond=0)
                if (nyc_datetime > start) & (nyc_datetime < end) :
                    time.sleep((end-nyc_datetime).seconds)
                    
                #Halt before market close    
                nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
                start = nyc_datetime.replace(hour=15, minute=30, second=0,microsecond=0)
                end = nyc_datetime.replace(hour=16, minute=5, second=0,microsecond=0)
                if (nyc_datetime > start) & (nyc_datetime < end) :
                    time.sleep((end-nyc_datetime).seconds)
            
            #Check CPU temperature
            CPU = float(re.findall(r"[-+]?\d*\.\d+|\d+",MacTmp.CPU_Temp())[0])
            print('\nCurrent CPU temperature is %a °C\n' %CPU)  # To get CPU Temperature
            if (CPU > 95) :
                self.CPU_high_counter += 1
                if self.threads > 1 :
                    print('\nCPU temperature too high, dropping down a thread\n')
                    self.threads -= 1
                    time.sleep(2)
                else :
                    print('\nCPU temperature too high, at minimum thread level\n')
                    self.threads = 1
                if (self.CPU_high_counter>15) :
                    print('\nCPU temperature is holding too high, sleeping 5 minutes\n')
                    time.sleep(60*5)
            
            self.mode = random.randint(1,5)
            #self.mode = 1
            print('\nModel creation mode : %d\n' %self.mode)
            self.predict = random.sample(self.possibilities, 1)[0]
            self.use_n = random.randint(1, 5)
            self.use_stocks()
            self.selector()
            start_m = datetime.now()
            self.prep()
            self.make_model()
            stop_m = datetime.now()
            time_m += (stop_m - start_m).seconds
            
        t_n = round(time_m/number_of_models,2)
        print('\n{} seconds needed for {} models, {} seconds per model\n'.format(time_m, number_of_models, t_n))
        df = pd.DataFrame({'date': [datetime.now().strftime('%d-%m-%y')], 'n_models': [number_of_models], 'time' : [time_m], 't_n' : [t_n]})
        df.to_csv('./log/model_making/time.csv')
        
if __name__ == "__main__":
    main()

