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
warnings.simplefilter(action = 'ignore')

def main():
    print('\n Running model Evaluation \n')
    market().execute()
         
class market :
    def __init__(self) :
        self.verify_features_store()
        self.path = os.getcwd()
        self.price_data = pd.read_csv('./data/features_store.csv',',')
        
        # Halt during pre-trading times
        nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
        start = nyc_datetime.replace(hour=7, minute=20, second=0,microsecond=0)
        end = nyc_datetime.replace(hour=9, minute=30, second=0,microsecond=0)
        if (nyc_datetime > start) & (nyc_datetime < end) :
            time.sleep(5400)
        
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
        price_data = pd.read_csv('./data/features_store.csv',',')
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        today = datetime.today()
        if price_data['Date'].iloc[0] < today - timedelta(days=10) :
            error_message = "The features store has not been updated for ten or more days, this is evaluated as a fatal error as it can lead to incorrect models, please update features store"
            self.error_handling(error_message)
                    
    def files (self) :
        self.all = glob.glob(os.getcwd() + '/models/*.{}'.format('csv'))
        
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
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        use = self.use 
        dates = price_data['Date']
        dates = dates.drop_duplicates()
        dates = dates.tolist()
        
        end_date = dates[self.test * 5]
        if self.test == 0 :
            intermediate = dates[100]
        else : 
            intermediate = dates[(self.test + 1) * 5]    
        intermediate = intermediate.strftime('%Y - %m - %d')
        end_date = end_date.strftime('%Y - %m - %d')
        today = self.price_data['Date'].iloc[0] 
        
        price_data['Date'] = pd.to_datetime(price_data['Date']) 
        price_data = price_data.loc[(price_data['Date'] < today)]
        
        print ('\n Testing from ', intermediate, 'to', end_date, ':')
        
        self.price_data_train = price_data[price_data['stock'].isin(use)].loc[(price_data['Date'] < intermediate)]
        self.price_data_test = price_data[price_data['stock'] == self.predict].loc[(price_data['Date'] < end_date) & (price_data['Date'] >= intermediate)]
        
        self.y_train = self.price_data_train['delta_class'].tolist()
        self.y_test = self.price_data_test['delta_class'].tolist() 
        
        if self.use_weights == 1 :
            self.make_weights()
        else :   
            self.weights = np.ones(len(self.y_train))
            
        record = pd.read_csv('./data/model_features.csv')
        self.features_name = record[self.model.replace(".csv", "")].dropna().tolist()
        self.features_name = [x for x in self.features_name if x in price_data.columns]
        self.X_train = self.price_data_train[self.features_name]
        self.X_test = self.price_data_test[self.features_name]

        return 
    
    def make_weights(self) :
        links = pd.read_csv('./data/stock_links.csv')
        weight = self.price_data_train[['stock','delta']]
        for stock in self.use :
            weight['stock'].loc[weight['stock'] == stock] = links[self.predict].loc[links['index'] == stock].values[0] * (2/100)
        
        weight['delta'].loc[weight['delta'] > 1] = 2
        weight['delta'].loc[weight['delta'] < -1] = 2
        weight['delta'].loc[weight['delta'] != 2] = 1
        
        self.weights = weight['delta'] + weight['stock']
        self.weights = self.weights.tolist()
        
    def record(self):
        record = pd.read_csv('./data/record_model.csv')
        today = datetime.today()
        rec = pd.DataFrame({'Date' : today.strftime('%Y - %m - %d'), 'Model_name' : self.model.replace(".csv", ""), 
                            'Stock': self.predict, 'Used' : str(self.use), 
                            'Parameters' : str(list(self.parameters.items())), 'Long_accuracy' : [self.accuracy], 
                            'Long_AP_score' : [self.AP], 'Long_ROC_AUC': [self.ROC], 'Long_Market_performance' : [self.market_per], 
                            'Long_Model_performance' : [self.model_per], 'Weekly_accuracy' : [self.accuracy_short], 
                            'Weekly_AP_score' : [self.AP_short], 'Weekly_ROC_AUC': [self.ROC_short],
                            'Trade_accuracy' : [self.accuracy_trade], 'Weekly_Trade_accuracy' : [self.accuracy_short_trade],
                            'Days_traded_long' : [self.days_long], 'Weekly_Days_traded' : [self.days_weekly],
                            'Model_level' : [self.model_level], 'Status' : [self.test]})
        
        record = record.append(rec, ignore_index=True)
        record.to_csv('./data/record_model.csv', index = False)

    def money(self, variations) :
        
        if variations == [] : 
            variation = self.price_data_test['delta'].values.tolist()
        else :
            variation = variations
            
        variation = np.array(variation)/100
        prediction = self.y_pred
        
        market = [1]
        account_long = [1]
        account_short = [1]
        account_both = [1]
        cost = 0
        
        for i in range (len(variation)):
            if (prediction[i]>0):
                account_long.append(account_long[-1] + account_long[-1] * (variation[i] - cost))
                account_short.append(account_short[-1])
                account_both.append(account_both[-1]+account_both[-1] * (variation[i] - cost))
            elif (prediction[i]<0) :
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
        self.model = model
        self.predict = hold[0]
        self.use = hold[2:-1]
        self.n_features = int(float(model.split('-')[1]))
        self.use_weights = int(float(model.split('-')[2]))
        
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
        level = 70
        thresholds = np.linspace(0.5, 1, num = 101)
        accuracy = 0
        n = 0
        accuracies = []
        accuracy = 0
        
        while accuracy < level :
            pred_level = []
            true_level = []
            top = round(thresholds[n],2)
            bottom = round(1 - thresholds[n],2)
            i = 0
            for prob in self.proba :
                if prob < bottom :
                    pred_level.extend([-1])
                    true_level.extend([self.y_test[i]])
                if prob > top :
                    pred_level.extend([1])
                    true_level.extend([self.y_test[i]])
                i += 1
            accuracies.extend([round(accuracy_score(true_level, pred_level, normalize = True) * 100.0,2)])
            accuracy = accuracies[-1]
            if n == len(thresholds) - 1:
                break
            else :
                n += 1

        return thresholds[n]
    
    def ml_model(self):
        train_data = lgb.Dataset(self.X_train,label=self.y_train, weight=self.weights)
        num_round = 10
        self.lgbm = lgb.train(self.parameters,train_data,num_round, verbose_eval=False)
        self.y_probs = self.lgbm.predict(self.X_test)
        
        self.y_pred = []
        for i in range(len(self.y_probs)):
            if self.y_probs[i]>=.5 :
               self.y_pred.append(1)
            else :  
               self.y_pred.append(-1)
               
    def best_model_purging(self): 
        record = pd.read_csv('./data/record_model.csv')
        best_models = glob.glob(os.getcwd() + '/Best_models/*.{}'.format('csv'))
        n_best_models = len(best_models)
        Weekly_ROC_AUC = []
        if n_best_models > 50:
            for model in best_models :
                model_search = model.split('/')[-1].replace('.csv','')
                Weekly_ROC_AUC.append(record[record['Model_name'] == model_search].iloc[-1]['Weekly_ROC_AUC'])
                
            quality = pd.DataFrame({'Model':best_models, 'ROC': Weekly_ROC_AUC})
            quality = quality.sort_values(by=['ROC'], ascending=True)
            
            n_to_delete = 5
            to_delete = quality['Model'].to_list()[:n_to_delete]    
            for model in to_delete :
                os.remove(model)
                           
    def test_all(self) :
        self.files()
        features = pd.read_csv('./data/model_features.csv')
        for model in self.all :
            accuracy = []
            gain = []
            market = []
            print(model)
            self.layout(model)
            self.test = 0
            self.prep()
            print(self.parameters)
            self.ml_model ()
            self.market_per = self.money([])[1]
            self.model_per = self.money([])[0]
            self.metric(self.y_probs)
            self.accuracy = accuracy_score(self.y_test, self.y_pred, normalize = True) * 100.0
            self.accuracy_trade = accuracy_score(self.y_true_trade, self.y_trade, normalize = True) * 100.0
            self.AP = average_precision_score(self.y_test, self.y_probs) * 100
            self.ROC = roc_auc_score(self.y_test, self.y_probs) * 100
            feature_imp = pd.Series(self.lgbm.feature_importance(), index=self.X_test.columns).sort_values(ascending=False)
            self.days_long = len(self.y_trade)

            print('\nModel accuracy over 100 days is :', np.round(self.accuracy,2))
            print('Model accuracy over 55 % over 100 days is :', np.round(self.accuracy_trade,2))
            print ('ROC AUC over 100 days is :', np.round(self.ROC,2))
            print ('Average precision score over 100 days is :', np.round(self.AP,2))
            print ('Market gain over 100 days is :', np.round(self.market_per,2), '\n Model gain over 100 days is :', np.round(self.model_per,2), '\n')
            print('Traded', np.round(int(self.days_long),2), 'days')
            
            self.prediction = []
            correct = []
            self.proba = []
            variations = []
            for i in range (1,30):
                
                self.test = i
                self.prep()
                if len(self.y_test)>0:
                    self.ml_model ()
                    self.prediction.extend(self.y_pred)
                    correct.extend(self.y_test)
                    self.proba.extend(self.y_probs)
                    variations.extend(self.price_data_test['delta'].tolist())
                    print('Completed :', np.round(100*i/30,2), '%')
            
            self.y_test = correct
            self.metric(self.proba)
            self.accuracy_short = accuracy_score(correct, self.prediction, normalize = True) * 100
            self.accuracy_short_trade = accuracy_score(self.y_true_trade, self.y_trade, normalize = True) * 100.0
            self.AP_short = average_precision_score(correct, self.proba) * 100
            self.ROC_short = roc_auc_score(correct, self.proba) * 100
            self.days_weekly = len(self.y_trade)
            self.model_level = self.threshold()
            
            print ('\nThe metrics over all periods considered are : ')
            print('Model accuracy is :', np.round(self.accuracy_short,2))
            print('Model accuracy over 55% is :', np.round(self.accuracy_short_trade,2))
            print('Model ROC AUC is :', np.round(self.ROC_short,2))
            print('Average precision score is :', np.round(self.AP_short,2))
            print('Traded', np.round(int(self.days_weekly),2), 'days')
            print('Model trade threshold is', np.round(self.model_level,2))
            
            
            n_best_models = len(glob.glob(os.getcwd() + '/Best_models/*.{}'.format('csv')))
            pass_threshold = 61 + (n_best_models/40)
            print('\n Using pass threshold of', pass_threshold)
            if  (self.ROC_short > pass_threshold) &  (self.accuracy_short_trade > pass_threshold) :
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
        self.best_model_purging()
        stop = datetime.now()
        print('\n Time for model evaluation : ', (stop - start))

if __name__ == "__main__":
    main()

