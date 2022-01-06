#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 20:51:28 2021

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
import xgboost as xgb
import lightgbm as lgb 
import pytz
from xgboost.sklearn import XGBClassifier
from email_updates_error import *
warnings.simplefilter(action = 'ignore')

def main():
    trade().execute()
    
class trade :
    def __init__(self) :
        """Built-in method to inialize the global values for the module and do intial checks

        Attributes
        -----------
        `self.verify_features_store()` : Function call
            Initial function call to the class function used to check receceny of data"
        `self.price_data ` : pandas DataFrame
            Latest version of the features store"
        `self.path` : str
            Get current working folder path.
        """
        
        files = glob.glob('*.{}'.format('csv'))
        self.price_data = pd.read_csv('./data/features_store.csv')
        self.price_data = self.price_data.dropna(axis=1, thresh=int(np.shape(self.price_data)[0]*0.95))
        self.flag_error = 0
        self.path = os.getcwd()
        data = pd.DataFrame({'Date' : [], 'Products' : [], 
                    'Probabilities': [], 'Model_level_p': [], 'Model_level_n': [],'Model_performance': []}) 
        data.to_csv('./data/trade_data.csv', index = False)
          
        return
    
    def error_handling(self, error_message) :
        """Class function used to handle errors, will send email using the imported error function from 'email_updates_error'

        Parameters
        ----------
        `error_message` : Information about the error
        """
        
        today = datetime.today()
        error_location = "model.py"
        error_report = pd.DataFrame({'Date' : today.strftime('%Y - %m - %d'), 'Code_name' : [error_location], 'Message' : [error_message]})
        error_report = error_report.set_index('Date')
        error(error_report)
        print(error_message)
        print('Sub-code sleeping until user kill')
        time.sleep(10000000)
              
    def cp_models(self) :
        """Class function used to copy best models from './Best_models' to './models_in_use'. This is to avoid conflict between model test and model usage

        Attributes
        -----------
        `self.modellist` : list
            list of all models available for trading"
        """
        
        src_dir = os.getcwd() + '/Best_models/'
        dest_dir = os.getcwd() + '/models_in_use/'
        os.chdir(dest_dir)
        files = glob.glob('*.{}'.format('csv'))
        for filename in files:
            os.remove(filename)
        os.chdir('..')
        os.chdir(src_dir)
        files = glob.glob('*.{}'.format('csv'))
        os.chdir('..')
        for filename in files:
            shutil.copy(src_dir + filename, dest_dir)
        self.modellist = files 
    
    def prep(self) :
        """Class function used to prepare the data for a model training, will select data for the used and predicted stock as well as the correct date period

        Attributes
        -----------
        `self.y_train` : list
            Represents the outcomes for all the train data"
        `self.features_name` : list
            Features to be used for the given model, only availbale features will be used, this can lead to less features used and weaker predictions"
        `self.X_train` : Pandas DataFrame
            Return train data after features selection"
        `self.X_test` : Pandas DataFrame
            Data used for the prediction, the test is analogeous to train/test, but here represents the data fot the final unknown outcome day"
        """
        record = pd.read_csv('./data/model_features.csv') # Read features record to extract features specific to the model in use
        price_data = self.price_data
        use = self.use 
        price_data['Date'] = pd.to_datetime(price_data['Date']) # Convert date column to datetime format for subsequent date selection 
        today = price_data['Date'][price_data['stock']==self.predict].iloc[0] # Extract last date in features store, which sould correspond to todays date                     
        print('\nTodays features date is %s' %today.date())
        
        if today.date() < (datetime.today()).date():
            self.flag_error = 1
            print(self.predict, 'is not up to date, it will be skipped \n')
        else :
            self.price_data_train = price_data[price_data['stock'].isin(use)].loc[(price_data['Date'] < today)]    
            price_data_test = price_data[price_data['stock'] == self.predict].loc[(price_data['Date'] == today)] # Extract data for stock to be predicted
            
            self.y_train = self.price_data_train['delta_class'].tolist() # Extract all outcomes for training
                
            self.features_name = record[self.model.replace(".csv", "")].dropna().tolist() 
            self.features_name = [x for x in self.features_name if x in price_data.columns] # Account for potential feature store modifications, WARNING this can lead weaker predictions
            self.X_train = self.price_data_train[self.features_name] # Load as class attribute the train set
            self.X_test = price_data_test[self.features_name] # Load as class attribute todays data for the stock to predict
            print('\n The folowing stock will predicted using the adjacent features :')
            print(self.X_test)
            print('\n')

        return 
    
    def model_prediction(self) :
        """Class function used to make prediction using model (lgbm_light) with paramaters supplied by model caracteristics

        Attributes
        -----------
        `self.lgbm` : ml_model
            Trained model"
        `self.prediction` : float
            Probability for todays outcome P < 0.5 : 0, P > 0.5 : 1"
        """
        y_train = self.y_train
        X_train = self.X_train
        
        train_data = lgb.Dataset(self.X_train,label=self.y_train)
        num_round = 100
        self.lgbm = lgb.train(self.parameters,train_data,num_round, verbose_eval=False)
        self.prediction = self.lgbm.predict(self.X_test)
        
        return
    
    def pre_trade_data(self) :
        """Class function used to create 'trade_data.csv' file with all predictions, this is pre-trade, before probability levels are applied.

        """
        
        data = pd.read_csv('./data/trade_data.csv')
        today = datetime.today()
        df = pd.DataFrame({'Date' : today.strftime('%Y - %m - %d'), 'Products' : self.predict, 
                    'Probabilities': [self.prediction[0]], 'Model_level_p': [self.model_level_p], 'Model_level_n': [self.model_level_n],
                    'Model_performance': [self.model_performance]})
        data = data.append(df, ignore_index=True)
        data.to_csv('./data/trade_data.csv', index = False)
        
    def distribution(self) :
        df = pd.read_csv('./data/to_trade.csv').dropna()
        trade_probability = df.groupby(['Products']).mean()['Prob_distance']
        Model_performance = df.groupby(['Products']).mean()['Model_performance']
        Probability = df.groupby(['Products']).mean()['Probability']
        Prob_distance = df.groupby(['Products']).mean()['Prob_distance']
        df = df.groupby(['Products']).sum()
        df['Probability'] = Probability
        df['Balanced_probability'] = trade_probability + 0.5
        df['Model_performance'] = Model_performance
        df['Prob_distance'] = Prob_distance
        
        if (not len(glob.glob('./data/record_traded.csv'))) or (len(list(set(pd.read_csv('./data/record_traded.csv')['Traded'])))<18):

            df['K%'] = df['Balanced_probability'] - (1-df['Balanced_probability'])/df['Model_performance']
            df['K%'] = df['K%']/df['K%'].sum()
            df = df.round(4)
            df['Side'][df['Side']<0] = -1
            df['Side'][df['Side']>0] = 1
            df['Side'][df['Side']==0] = np.nan
            
        else :
            record = pd.read_csv('./data/record_all_predictions.csv')
            record['rate'] = record['Prediction']*record['Outcome']
            record['rate'][record['rate']<0] = 0
            del record['Probability']
            record = record.groupby(['Traded']).mean()
            df = pd.merge(df, record, left_index=True, right_index=True)

            df['K%'] = df['Balanced_probability'] - (1-df['Balanced_probability'])/df['rate']
            df['K%'] = df['K%']/df['K%'].sum()
            
        df = df.round(4)
        df['Side'][df['Side']<0] = -1
        df['Side'][df['Side']>0] = 1
        df['Side'][df['Side']==0] = np.nan
        df['K%'][df['K%']>0.1] = 0.15
        df['K%'][df['K%']<0.01] = np.nan
        df = df.reset_index().rename({'index': 'Products'}, axis=1)
        df = df.dropna()
        df.to_csv('./data/to_trade.csv',index = False)
            
            
        return df
        
        
    def trade_data(self) :
        """Class function used to create 'to_trade.csv' file with all predictions to be traded. 
        Function does two selections :
            - Selection based on random noise, if the mean probability of all models for a stock is between 0.45 & 0.55 
            then the overall prediction is considered as random and will not be traded.
            - Selection based on the threshold for each model. (Evaluated from model_evaluate.py)
        Function returns the trade information in 'to_trade.csv' for further use.
        """
        
        data = pd.read_csv('./data/trade_data.csv')
        stocks = data['Products'].tolist()
        stocks = list(set(stocks))
        today = datetime.today()
        to_trade = pd.DataFrame({'Products' : [], 
                    'Prob_distance': [], 'Side': [],
                    'Probability' : [], 'Model_performance': []})
        
        for stock in stocks:
            data_stock = data[data['Products'] == stock]
            probability = data_stock['Probabilities'][data_stock['Products'] == stock].tolist() 
            level_p = data_stock['Model_level_p'][data_stock['Products'] == stock].tolist()
            level_n = data_stock['Model_level_n'][data_stock['Products'] == stock].tolist()
            performance = data_stock['Model_performance'][data_stock['Products'] == stock].tolist()
            
            side = np.array([])
            prob_distance = np.array([])
            product = np.array([])
            prob = np.array([])
            model_performance = np.array([])
            for i in range(len(probability)) :
                if (probability[i] > 0.8) :
                    side = np.append(side, 1)
                    prob_distance = np.append(prob_distance, probability[i] - level_p[i])
                    product = np.append(product, stock)
                    prob = np.append(prob, probability[i])
                    model_performance = np.append(model_performance, 0.01*performance[i]/(1-performance[i]*0.01))
                if (probability[i] > 0.8) :
                    side = np.append(side, -1)
                    prob_distance = np.append(prob_distance, (1-probability[i]) - level_n[i])
                    product = np.append(product, stock)
                    prob = np.append(prob, probability[i])
                    model_performance = np.append(model_performance, 0.01*performance[i]/(1-performance[i]*0.01))
             
            df = pd.DataFrame({'Products' : product, 
                    'Prob_distance': prob_distance, 'Side': side, 'Probability' : prob, 'Model_performance' : model_performance})
            to_trade = to_trade.append(df, ignore_index=True)
        
        to_trade.to_csv('./data/to_trade.csv', index = False)
        
                        
    def model_info(self, model) :
        """Class function used to make prediction using model (lgbm_light) with paramaters supplied by model caracteristics

        Attributes
        -----------
        `self.model_level` : ml_model
            Trained model"
        `self.prediction` : float
            Probability for todays outcome P < 0.5 : 0, P > 0.5 : 1"
        """
        
        model = model.replace(".csv", "")
        record = pd.read_csv('./data/record_model.csv')
        record = record.drop_duplicates(subset=['model_name'], keep='last')
        record = record[record['model_name'] == model]
        self.model_level_p = np.round(record['model_level_live_p'].iloc[0],2)
        self.model_level_n = np.round(record['model_level_live_n'].iloc[0],2)
        self.model_performance = np.round(record['trade_accuracy_live'].iloc[0],2)

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
        parameters = pd.read_csv('./models_in_use/' + model)
        model = model.split('/')[-1]
        hold = ''.join([i for i in model if not i.isdigit()])
        hold = hold.split('-')
        if '' in hold: hold.remove('')
        hold = [s.replace(".csv", "") for s in hold]
        parameters = parameters[model.replace(".csv", "")].iloc[0]
        parameters = eval(parameters.replace('nan','np.nan'))
        self.parameters = dict(parameters)
        self.parameters['num_threads'] = 11
        self.model = model
        self.predict = hold[0]
        self.use = hold[1:-1]
    
    def record (self) :
        """Class function used to record the days trades and outcomes for subsequent statistics, data is pulled from yahoo finance

        """
        if not len(glob.glob('./data/record_traded.csv')):
            record = pd.DataFrame({'Date' : [], 
                        'Traded': [], 'predictions': [],
                        'outcome': [], 'Delta': [], 'Probability': [], 
                        'Prob_distance': []})   
            record.to_csv('./data/record_traded.csv', index = False)
            
        if not len(glob.glob('./data/record_all_predictions.csv')):
            record = pd.DataFrame({'Date' : [], 
                        'Traded': [], 'predictions': [],
                        'outcome': [], 'Delta': [], 'Probability': []})  
            record.to_csv('./data/record_all_predictions.csv', index = False)
        
        if not len(glob.glob('./data/account.csv')):
            account = pd.DataFrame({'Date' : [], 'AM' : [], 'PM' : [], 
                                    'Trade_value' : [], 'Change_account_%' : []})
            account.to_csv('./data/account.csv', index = False) 
            
        today = datetime.today()
        
        if np.size(pd.read_csv('./data/to_trade.csv'))>0 :
            record_traded = pd.read_csv('./data/record_traded.csv')
            traded = pd.read_csv('./data/to_trade.csv')
            stocks = list(set(traded['Products'].tolist()))
    
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
                
                df = pd.DataFrame({'Date' : today.strftime('%Y - %m - %d'), 
                            'Traded': st, 'Prediction': pred,
                            'Outcome': outcomes, 'Delta': deltas, 'Probability': prob, 
                            'Prob_distance': prob_dist})   
                record_traded = record_traded.append(df, ignore_index=True)
            record_traded.to_csv('./data/record_traded.csv', index = False)
        
        if np.size(pd.read_csv('./data/trade_data.csv'))>0 :
            record_all = pd.read_csv('./data/record_all_predictions.csv')
            traded = pd.read_csv('./data/trade_data.csv')
            traded['Side'] = traded['Probabilities']
            traded['Side'].loc[traded['Side'] > 0.5] = 1
            traded['Side'].loc[traded['Side'] < 0.5] = -1
            stocks = list(set(traded['Products'].tolist()))
    
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
                
                df = pd.DataFrame({'Date' : today.strftime('%Y - %m - %d'), 
                            'Traded': st, 'Prediction': pred,
                            'Outcome': outcomes, 'Delta': deltas, 'Probability': prob}) 
                
                record_all = record_all.append(df, ignore_index=True)
            record_all.to_csv('./data/record_all_predictions.csv', index = False)        
        
        if np.size(pd.read_csv('./data/to_trade.csv'))>0 :
            account = pd.read_csv('./data/account.csv')
            account_value = pd.read_csv('./data/account_value.csv')
            account = account.append(account_value, ignore_index=True)
            account.to_csv('./data/account.csv', index = False)  
        
        print(account.tail(1))
        return
           
    def execute (self) :
        """Class function used to execute days trading system using all functions above.

        Code Calls
        -----------
        `python alpaca_trading.py` : python code
            Code used to execute the trades predicted"
        `python email_updates_evening.py` : python code
            Code used to send update email of outcome of days trades, data pulled from 'record.csv'"
        """
        
        print('\nIn trading system, initiating model training and prediction\n')
        self.cp_models()
        for mdl in self.modellist :
            self.flag_error = 0
            print(mdl)
            self.layout(mdl)
            print('Model for : ', self.model)
            self.prep()
            if not self.flag_error :
                self.model_prediction()
                self.model_info(mdl)
                self.pre_trade_data()
        
        print('\nModel training and prediction --- Completed\n')
        if np.size(pd.read_csv('./data/trade_data.csv'))>0 :
            self.trade_data()

            if np.size(pd.read_csv('./data/to_trade.csv'))>0 :
                self.distribution()
                print('\nExecuting alpaca trading \n')
                os.system("python alpaca_trading.py")
            else :
                print('\nNo trades above set probability level\n')

            nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
            close = nyc_datetime.replace(hour=16, minute=0, second=0,microsecond=0)
            if nyc_datetime < close:
                difference = close - nyc_datetime
                print('\nWaiting for market close', round((difference.seconds/60)/60,2), 'hours\n')
                time.sleep(difference.seconds+60)
        
        else : 
            print('\nEmpty trade sheet - No models or none up to date\n')
            
        nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
        close = nyc_datetime.replace(hour=16, minute=0, second=0,microsecond=0)
        if nyc_datetime < close:
            difference = close - nyc_datetime
            print('\nWaiting for market close', round((difference.seconds/60)/60,2), 'hours\n')
            time.sleep(difference.seconds+60)
         
        self.record()
        os.system('python email_updates_evening.py')

        return 
    
if __name__ == "__main__":
    main()
