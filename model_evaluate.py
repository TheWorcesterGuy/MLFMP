#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 00:55:28 2021

@author: christian
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import glob
import sys
import warnings
import os, signal
import shutil
import ast
import xgboost as xgb
import lightgbm as lgb 
from xgboost.sklearn import XGBClassifier
import random
from datetime import datetime, timedelta
import pytz
import time
from email_updates_error import *
from clean import *
import re
import MacTmp
warnings.simplefilter(action = 'ignore')

def main():
    clean_features()
    print('\n Running model Evaluation \n')
    market().execute()
         
class market :
    def __init__(self) :
        self.verify_features_store()
        self.path = os.getcwd()
        self.price_data = pd.read_csv('./data/features_store.csv',',')
        self.price_data = self.price_data.dropna(axis=1, thresh=int(np.shape(self.price_data)[0]*0.95))
        self.CPU_high_counter = 0
        self.threads = 6
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
        
    def error_handling(self, error_message) :
        today = datetime.today()
        error_location = "model.py"
        error_report = pd.DataFrame({'Date' : today.strftime('%Y - %m - %d'), 'Code_name' : [error_location], 'Message' : [error_message]})
        error_report = error_report.set_index('Date')
        error(error_report)
        print(error_message)
        print('Sub-code sleeping until user kill')
        time.sleep(10000000)
              
    def verify_features_store(self):
        price_data = pd.read_csv('./data/features_store.csv')
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        today = datetime.today()
        if price_data['Date'].iloc[0] < today - timedelta(days=50) :
            error_message = "The features store has not been updated for ten or more days, this is evaluated as a fatal error as it can lead to incorrect models, please update features store"
            self.error_handling(error_message)
        
        record_model_available = len(glob.glob('./data/record_model.csv'))
        if not record_model_available :
            record = pd.DataFrame(columns=['date', 'model_name', 'stock', 'used', 'parameters', 'accuracy_test', 'ROC_test','trade_accuracy_test','days_traded_test', 'model_level_test_n','model_level_test_p', 'market_performance_test','model_performance_test', 'accuracy_live', 'ROC_live','trade_accuracy_live', 'days_traded_live','model_level_live_n','model_level_live_p', 'market_performance_live','model_performance_live','status'])
            record.to_csv('./data/record_model.csv', index = False)  
            
        if len(glob.glob('./models/*.csv')) == 0 :
            os.system('rm ./data/model_features.csv')
            os.system('rm -r ./Best_models')
            os.system('mkdir ./Best_models')
        
    def files (self) :
        self.all = glob.glob(os.getcwd() + '/models/*.{}'.format('csv'))
        random.shuffle(self.all)
        dir = os.getcwd() + '/Best_models'
        filelist = glob.glob(os.path.join(dir, "*"))
        
        for f in filelist:
            name = f.split('/')[-1]
            self.all = [ x for x in self.all if name not in x ]
            
        nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
        start = nyc_datetime.replace(hour=17, minute=0, second=0,microsecond=0)
        end = nyc_datetime.replace(hour=17, minute=30, second=0,microsecond=0)
        if (nyc_datetime > start) & (nyc_datetime < end) :
            for f in filelist :
                os.remove(f)
                
    def prep(self) :
        price_data = self.price_data
        use = self.use 
        price_data['Date'] = pd.to_datetime(price_data['Date']) 
        dates = price_data['Date']
        dates = dates.drop_duplicates()
        #dates = dates.iloc[100:]
        today = dates.iloc[0]
        last_date = dates.iloc[5*(self.live) + 1*(self.test>0)*(105 + 5*self.test)]
        first_date = dates.iloc[1*(self.live>0)*5*(self.live+1) + 1*(self.test>0)*(105 + 5*(self.test+1))] # We chose 200 as upper limit
        price_data = price_data.loc[(price_data['Date'] < today)]
        self.price_data_train = price_data[price_data['stock'].isin(use)].loc[(price_data['Date'] < first_date)]
        self.price_data_test = price_data[price_data['stock'] == self.predict].loc[(price_data['Date'] < last_date)]
        self.price_data_test = self.price_data_test.loc[(price_data['Date'] >= first_date)]
        self.y_train = self.price_data_train['delta_class'].tolist()
        self.y_test = self.price_data_test['delta_class'].tolist() 
        
        record = pd.read_csv('./data/model_features.csv')
        self.features_name = record[self.model.replace(".csv", "")].dropna().tolist()
        self.features_name = [x for x in self.features_name if x in price_data.columns]               
        self.X_train = self.price_data_train[self.features_name]
        self.X_test = self.price_data_test[self.features_name]
            
        return 
        
    def record(self):
        record = pd.read_csv('./data/record_model.csv')
        today = datetime.today()
        rec = pd.DataFrame({'date' : today.strftime('%Y - %m - %d'), 'model_name' : self.model.replace(".csv", ""), 
                            'stock': self.predict, 'used' : str(self.use), 
                            'parameters' : str(list(self.parameters.items())), 
                            'accuracy_test' : [np.round(self.accuracy_test,2)], 'ROC_test': [np.round(self.ROC_test,2)], 
                            'trade_accuracy_test' : [np.round(self.accuracy_test_trade,2)], 'days_traded_test' : [self.days_test], 
                            'model_level_test_p' : [np.round(self.model_level_test_p,2)], 'model_level_test_n' : [np.round(self.model_level_test_n,2)],
                            'market_performance_test' : [np.round(self.market_test,2)], 'model_performance_test' : [np.round(self.account_test,2)], 
                            'accuracy_live' : [np.round(self.accuracy_live,2)], 'ROC_live': [np.round(self.ROC_live,2)], 
                            'trade_accuracy_live' : [np.round(self.accuracy_live_trade,2)], 'days_traded_live' : [self.days_live], 
                            'model_level_live_p' : [np.round(self.model_level_live_p,2)], 'model_level_live_n' : [np.round(self.model_level_live_n,2)],
                            'market_performance_live' : [np.round(self.market_live,2)], 'model_performance_live' : [np.round(self.account_live,2)], 
                            'status' : [self.test]})
        
        record = record.append(rec, ignore_index=True)
        record.to_csv('./data/record_model.csv', index = False)

    def money(self, variations, predictions) :
                    
        variation = np.array(variations)/100
        
        market = [1]
        account_long = [1]
        account_short = [1]
        account_both = [1]
        cost = 0
        
        for i in range (len(variation)):
            if (predictions[i]>0):
                account_long.append(account_long[-1] + account_long[-1] * (variation[i] - cost))
                account_short.append(account_short[-1])
                account_both.append(account_both[-1]+account_both[-1] * (variation[i] - cost))
            elif (predictions[i]<0) :
                account_long.append(account_long[-1])
                account_short.append(account_short[-1] - account_short[-1] * (variation[i] + cost))
                account_both.append(account_both[-1] - account_both[-1] * (variation[i] + cost))
            else:
                account_long.append(account_long[-1])
                account_short.append(account_short[-1])
                account_both.append(account_both[-1])
        
            market.append(market[-1] + market[-1] * variation[i])
        
        return account_both[-1], market[-1]
    
    def layout(self, model) :
        """Class function used to load the model in use parameters from the model csv

        Attributes
        -----------
        `self.parameters` : dict
            ml_model marapmeters"
        `self.model` : str
            Name of the model"
        `self.predict` : str
            Stock to predict"
        `self.use` : list
            Stocks to use to reinforce train set"
        `self.n_features : list
            Number of features to use"
        `self.use_weights : int
            Indicate to system whether feature weights should be used or not, takes 0 or 1
        """
        
        parameters = pd.read_csv(model)
        model = model.split('/')[-1]
        hold = ''.join([i for i in model if not i.isdigit()])
        hold = hold.split('-')
        if '' in hold: hold.remove('')
        hold = [s.replace(".csv", "") for s in hold]
        parameters = parameters[model.replace(".csv", "")].iloc[0]
        parameters = eval(parameters.replace('nan','np.nan'))
        self.parameters = dict(parameters)
        self.parameters['num_threads'] = self.threads
        self.model = model
        self.predict = hold[0]
        self.use = hold[1:-1]
        
    def metric(self,probabilities) :
        
        self.y_trade = []
        self.y_true_trade = []
        counter = 0
        for prob in probabilities :
            if prob >= 0.55 :
                self.y_trade.append(1)
                self.y_true_trade.append(self.y_test[counter])
            elif prob <= 0.45 :
                self.y_trade.append(-1)
                self.y_true_trade.append(self.y_test[counter])
            counter += 1
            
    def threshold (self) :
        level = 0.7
        thresholds = np.linspace(0.5, 1, num = 1000)
        accuracy_n, accuracy_p = 0 , 0
        
        #Positive side
        n = 0
        while accuracy_p < level :
            pred_level = []
            true_level = []
            top = round(thresholds[n],5)
            i = 0
            for prob in self.proba :
                if prob > top :
                    pred_level.extend([1])
                    true_level.extend([self.y_test[i]])
                i += 1
            accuracy_p = round(accuracy_score(true_level, pred_level, normalize = True),2)
            level_p = thresholds[n]
            if n == len(thresholds) - 1:
                break
            else :
                n += 1
        
        #negative side
        n = 0
        while accuracy_n < level :
            pred_level = []
            true_level = []
            bottom = round(1 - thresholds[n],5)
            i = 0
            for prob in self.proba :
                if prob < bottom :
                    pred_level.extend([-1])
                    true_level.extend([self.y_test[i]])
                i += 1
            accuracy_n = round(accuracy_score(true_level, pred_level, normalize = True),2)
            level_n = thresholds[n]
            if n == len(thresholds) - 1:
                break
            else :
                n += 1
                
        if np.isnan(level_p):
            level_p = 1
        if np.isnan(level_n):
            level_n = 1
            
        return level_p, level_n
    
    def ml_model(self):
        train_data = lgb.Dataset(self.X_train,label=self.y_train)
        num_round = 100
        self.lgbm = lgb.train(self.parameters,train_data,num_round, verbose_eval=False)
        self.y_probs = self.lgbm.predict(self.X_test)
        
        self.y_pred = []
        for i in range(len(self.y_probs)):
            if self.y_probs[i]>=.5 :
               self.y_pred.append(1)
            else :  
               self.y_pred.append(-1)
                           
    def test_all(self) :
        
        self.files()            
        for model in self.all :
            
            #Check CPU temperature
            CPU = float(re.findall(r"[-+]?\d*\.\d+|\d+",MacTmp.CPU_Temp())[0])
            print('\nCurrent CPU temperature is %a °C\n' %CPU)  # To get CPU Temperature
            if (CPU > 95) :
                self.CPU_high_counter += 1
                if self.threads > 1 :
                    print('\nCPU temperature too high, dropping down a thread\n')
                    self.threads -= 1
                else :
                    print('\nCPU temperature too high, at minimum thread level\n')
                    self.threads = 1
                if (self.CPU_high_counter>15) :
                    print('\nCPU temperature is holding too high, sleeping 5 minutes\n')
                    time.sleep(60*5)  
                
            features = pd.read_csv('./data/model_features.csv')
            print(model)
            self.layout(model)
            print(self.parameters)

            self.prediction = []
            correct = []
            self.proba = []
            variations = []
            self.live = 0
            for i in range (1,31):
                self.test = i
                self.prep()
                if len(self.y_test)>0:
                    self.ml_model()
                    self.prediction.extend(self.y_pred)
                    correct.extend(self.y_test)
                    self.proba.extend(self.y_probs)
                    variations.extend(self.price_data_test['delta'].tolist())
                    #print('Completed :', np.round(100*i/30,2), '%')
            
            self.account_test, self.market_test = self.money(variations, self.prediction)
            self.y_test = correct
            self.metric(self.proba)
            self.accuracy_test = accuracy_score(correct, self.prediction, normalize = True) * 100
            if np.isnan(self.accuracy_test):
                self.accuracy_test = 0
            self.accuracy_test_trade = accuracy_score(self.y_true_trade, self.y_trade, normalize = True) * 100.0
            if np.isnan(self.accuracy_test_trade):
                self.accuracy_test_trade = 0
            self.ROC_test = roc_auc_score(correct, self.proba) * 100
            self.days_test = len(self.y_trade)
            self.model_level_test_p, self.model_level_test_n = self.threshold()
            
            print ('\nThe metrics over the test are : ')
            print('Model accuracy is :', np.round(self.accuracy_test,2))
            print('Buy and hold :', np.round(self.market_test,2), 'model :', np.round(self.account_test,2))
            print('Model accuracy over 55% is :', np.round(self.accuracy_test_trade,2))
            print('Model ROC AUC is :', np.round(self.ROC_test,2))
            print('Traded', np.round(int(self.days_test),2), 'days')
            print('Model trade threshold positive side is', np.round(self.model_level_test_p*100,2))
            print('Model trade threshold negative side is', np.round(self.model_level_test_n*100,2))
            
            self.prediction = []
            correct = []
            self.proba = []
            variations = []
            self.test = 0
            for i in range (1,21):
                self.live = i
                self.prep()
                if len(self.y_test)>0:
                    self.ml_model ()
                    self.prediction.extend(self.y_pred)
                    correct.extend(self.y_test)
                    self.proba.extend(self.y_probs)
                    variations.extend(self.price_data_test['delta'].tolist())
                    #print('Completed :', np.round(100*i/10,2), '%')
            
            self.account_live, self.market_live = self.money(variations, self.prediction)
            self.y_test = correct
            self.metric(self.proba)
            self.accuracy_live = accuracy_score(correct, self.prediction, normalize = True) * 100
            if np.isnan(self.accuracy_live):
                self.accuracy_live = 0
            self.accuracy_live_trade = accuracy_score(self.y_true_trade, self.y_trade, normalize = True) * 100.0
            if np.isnan(self.accuracy_live_trade):
                self.accuracy_live_trade = 0
            self.ROC_live = roc_auc_score(correct, self.proba) * 100
            self.days_live = len(self.y_trade)
            self.model_level_live_p, self.model_level_live_n = self.threshold()
            
            print ('\nThe metrics over the live test are : ')
            print('Model accuracy is :', np.round(self.accuracy_live,2))
            print('Buy and hold :', np.round(self.market_live,2), 'model :', np.round(self.account_live,2))
            print('Model accuracy over 55% is :', np.round(self.accuracy_live_trade,2))
            print('Model ROC AUC is :', np.round(self.ROC_live,2))
            print('Traded', np.round(int(self.days_live),2), 'days')
            print('Model trade threshold positive side is', np.round(self.model_level_live_p*100,2))
            print('Model trade threshold negative side is', np.round(self.model_level_live_n*100,2))
            
            n_best_models = len(glob.glob(os.getcwd() + '/Best_models/*.{}'.format('csv')))
            pass_threshold = 58 + (n_best_models/40)
            percent_days = 10/100
            print('\n Using pass threshold of', np.round(pass_threshold,2))
            if  (self.ROC_test > pass_threshold) &  (self.ROC_live > pass_threshold) & \
                    (self.accuracy_test_trade > pass_threshold) & (self.accuracy_live_trade > pass_threshold) & \
                        (self.days_test > int(150*percent_days)) & (self.days_live > int(100*percent_days)) :
                self.test = 1
            else :
                self.test = 0
                             
            if (self.test == 1) :
                print('\n Passed test \n')
                print(model, '\n Will be selected \n')
                shutil.copyfile(os.getcwd() + '/models/' + self.model, os.getcwd() + '/Best_models/' + self.model)
                self.status = 1 
            else :
                print('\n Failed test \n')
                print(model, '\n Will be deleted \n')
                os.remove(model)
                self.status = 0
                features = features.drop([self.model.replace(".csv", "")], axis=1)
                        
            self.record()
            
            if len(features.columns) < 1 :
                os.remove('./data/model_features.csv')
            else :
                features.to_csv('./data/model_features.csv', index = False)
                
    def execute(self) :
        start = datetime.now()
        self.test_all()
        stop = datetime.now()
        print('\n Time for model evaluation : ', (stop - start))

if __name__ == "__main__":
    main()

