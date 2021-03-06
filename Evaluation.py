#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 00:06:09 2021

@author: christian
"""

from datetime import datetime
from alpaca_trade_api.rest import TimeFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
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
import random
warnings.simplefilter(action = 'ignore')

def main():
    print('\n Evaluating recorded models \n')

    #evaluation().charts()
    #evaluation().money()
    evaluation().account()
    #evaluation().models_quality()
    #evaluation().models_quality_trade()
    #evaluation().results_traded()
    #evaluation().results_predicted()
    #evaluation().history()
    #evaluation().model_count()
    #evaluation().parameter_evolution()
    evaluation().annualized_gain()
    #evaluation().prediction_control()
    
class evaluation :
    def __init__(self, save = False, verbose = True):
        self.save = save
        self.verbose = verbose
        record = pd.read_csv('./data/record_model.csv')
        #record.drop_duplicates(subset=['model_name', 'parameters'], keep='first')
        record['date'] = pd.to_datetime(record['date'])
        today = datetime.now()
        recent = today - timedelta(days=1)
        record = record[record['date'] > recent].dropna()
        self.record = record.dropna()
        #self.start = datetime(2021, 10, 1, 0, 0, 0, 0)
        self.start = datetime(2022, 1, 9, 0, 0, 0, 0)
        account = pd.read_csv('./data/account.csv')
        account['Date'] = pd.to_datetime(account['Date'])
        self.avoid = account[account['Trade_factor'] != 1]['Date']
        
        if self.verbose :
            print(record.mean())
        
    def charts (self):
        
        record = self.record
        record['date'] = pd.to_datetime(record['date'])
        if type(self.start) != int :
            record = record[record['date'] > self.start]
        percentage_days = 10
        percentage_days = percentage_days/100
        record = record[record['days_traded_test'] > int(150*percentage_days)]
        record = record[record['days_traded_live'] > int(100*percentage_days)]
        
        print('\n{} models today\n'.format(record.count().date))
        
        #normalise by number of days traded 
        record['trade_accuracy_live'] = record['trade_accuracy_live']
        record['model_performance_live'] = record['model_performance_live']/record['days_traded_live']
        
        
        #record = record[record['status'] > 0]
        accuracy_test = sorted((record['trade_accuracy_test']).to_list())[:-10]
        accuracy_live = (record['trade_accuracy_live']).to_list()
        ROC_test = sorted((record['ROC_test']).to_list())[:-10]
        
        mean_live_thr = []
        mean_live_perf = []
        std_live_thr = []
        for value in accuracy_test :
            mean_live_thr.append(np.mean(record['trade_accuracy_live'][record['trade_accuracy_test'] > value].to_list()))
            std_live_thr.append(np.std(record['trade_accuracy_live'][record['trade_accuracy_test'] > value].to_list()))
            mean_live_perf.append(np.mean(record['model_performance_live'][record['trade_accuracy_test'] > value].to_list()))
        
        mean_live_ROC_thr = []
        mean_live_ROC_perf = []
        std_live_ROC_thr = []
        for value in ROC_test :
            mean_live_ROC_thr.append(np.mean(record['trade_accuracy_live'][record['ROC_test'] > value].to_list()))
            std_live_ROC_thr.append(np.std(record['trade_accuracy_live'][record['ROC_test'] > value].to_list()))
            mean_live_ROC_perf.append(np.mean(record['model_performance_live'][record['ROC_test'] > value].to_list()))
        
        if self.verbose :
            print('mean_test', np.round(np.mean(accuracy_test),2))
            print('mean_live', np.round(np.mean(accuracy_live),2))
        
        plt.title('Mean live accuracy over test accuracy threshold (normalised by days traded)', fontsize=10)
        plt.xlabel('Test accuracy threshold', fontsize=8)
        plt.ylabel('Mean live accuracy above threshold (normalised by days traded)', fontsize=8)
        plt.plot(accuracy_test, mean_live_thr, 'o')
        #plt.yscale('log')
        if self.save :
            plt.savefig('./Images/Mean live accuracy over test accuracy threshold.png')
            plt.close()
        else :
            plt.show()
        
        plt.title('Mean live accuracy over test ROC threshold', fontsize=10)
        plt.xlabel('Test ROC threshold', fontsize=8)
        plt.ylabel('Mean live accuracy above threshold (normalised by days traded)', fontsize=8)
        plt.plot(ROC_test, mean_live_ROC_thr, 'o')
        #plt.yscale('log')
        if self.save :
            plt.savefig('./Images/Mean live accuracy over test ROC threshold.png')
            plt.close()
        else :    
            plt.show()
        
        # plt.title('Standard deviation of performance over threshold')
        # plt.xlabel('Test accuracy threshold')
        # plt.ylabel('standard deviation above threshold')
        # plt.plot(accuracy_test, std_live_thr, 'o')
        # plt.show()
        
        plt.title('Trading results over test accuracy threshold (normalised by days traded)', fontsize=10)
        plt.xlabel('Test accuracy threshold', fontsize=8)
        plt.ylabel('Live results over threshold (normalised by days traded)', fontsize=8)
        plt.plot(accuracy_test, mean_live_perf, 'o')
        #plt.yscale('log')
        if self.save :
            plt.savefig('./Images/Mean live trading results over accuracy threshold.png')
            plt.close()
        else :
            plt.show()
        
        plt.title('Trading results over test ROC threshold (normalised by days traded)', fontsize=10)
        plt.xlabel('Test ROC threshold', fontsize=8)
        plt.ylabel('Live results over threshold (normalised by days traded)', fontsize=8)
        plt.plot(ROC_test, mean_live_ROC_perf, 'o')
        #plt.yscale('log')
        if self.save :
            plt.savefig('./Images/Trading results over test ROC threshold.png')
            plt.close()
        else :
            plt.show()
        
    def money(self) :
        
        df = pd.read_csv('./data/record_model.csv')
        df['date'] = pd.to_datetime(df['date'])
        today = datetime.now()
        recent = today - timedelta(days=2)
        if type(self.start) != int :
            df = df[df['date'] > self.start]
        plt.figure()
        df = df.groupby(['accuracy_live']).mean().reset_index()
        plt.plot(np.array(df['accuracy_live']),np.array(df['model_performance_live']),'o')
        plt.axhline(y=1,color='k',linestyle='--',label='Theoretical gain limit')
        plt.axvline(x=50,color='r',linestyle='--',label='No predictif power limit')
        plt.title('Link between accuracy and performance')
        plt.xlabel('Live accuracy')
        plt.ylabel('Performance')
        plt.legend()
        plt.show()
        
    def account (self) :
        def gain(init,change):
            first = init
            for ch in change :
                init = init*(1+ch/100)
            return (init/first) - 1
        
        account = pd.read_csv('./data/account.csv').dropna()
        account['date'] = pd.to_datetime(account['Date'])
        if type(self.start) != int :
            account = account[account['date'] > self.start]

        plt.figure()
        plt.plot(account['Date'],account['AM'],'r', label='AM')
        plt.plot(account['Date'],account['PM'],'b', label='PM')
        for date in self.avoid.iloc[:-1] :
            plt.axvline(x=date.strftime("%Y - %m - %d"), color='y', linestyle='-')
        plt.axvline(x=self.avoid.iloc[-1].strftime("%Y - %m - %d"), color='y', linestyle='-',label='Days avoided')
        plt.xticks(rotation='vertical')
        plt.xlabel('Date')
        plt.ylabel('Account value (in $)')
        plt.legend()
        plt.xticks(rotation=45, fontsize=8)
        
        n_days_running = len(account) 
        account = account[~account['date'].isin(self.avoid)]
        n_days_traded = len(account)
        r_days_traded = n_days_traded/n_days_running
        
        gain = gain(account['AM'].iloc[0],account['Change_account_%'])
        gain_per_day = 100*gain/n_days_running
        annualised_gain = np.round(((1+gain_per_day/100)**(253) - 1) * 100,2)
        gain = round(gain*100,2)
        plt.title("Account value since {}\n Change in value of {} % - Average daily change {} %\n Annualized gain of {} %"\
                  .format(self.start.strftime("%d/%m/%Y"), gain, np.round(gain_per_day,2), annualised_gain))
        
        if self.save :
            plt.savefig('./Images/account.png')
            plt.close()
        else :
            plt.show()
        
        if self.verbose :
            print('Gain/Loss percent on account %a' %gain)
        
    def models_quality(self):
        record = pd.read_csv('./data/record_model.csv').dropna()
        #record = record.drop_duplicates(subset=['model_name','parameters'], keep='first')
        record = record.groupby(['date']).mean().reset_index()
        record['date'] = pd.to_datetime(record['date']) 
        if type(self.start) != int :
            record = record[record['date'] > self.start]
        plt.figure()
        plt.plot(record['date'],record['trade_accuracy_test'],'b', label='Accuracy test')
        plt.plot(record['date'],record['ROC_test'],'--b', label='ROC test')
        plt.plot(record['date'],record['trade_accuracy_live'],'r', label='Accuracy live')
        plt.plot(record['date'],record['ROC_live'],'--r', label='ROC live')
        plt.xticks(rotation=45, fontsize=8)
        plt.title("Quality of models since {}".format(self.start.strftime("%d/%m/%Y")))
        plt.xlabel('Date')
        plt.ylabel('Metric')
        plt.legend()
        if self.save :
            plt.savefig('./Images/models_quality.png')
            plt.close()
        else :
            plt.show()
            
    def models_quality_trade(self):
        record = pd.read_csv('./data/record_traded.csv').dropna()
        record['Date'] = pd.to_datetime(record['Date'])
        if type(self.start) != int :
            record = record[record['Date'] > self.start]
        record['status'] = record['Prediction']*record['Outcome']
        record['status'][record['status']<0] = 0
        record = record.groupby(['Date']).mean().reset_index()
        print('\nMean daily traded accuracy is {}'.format(np.round(100*np.mean(record['status']),2)))
        
        plt.figure()
        plt.plot(record['Date'],record['status']*100,'b', label='Daily accuracy traded')
        
        df = pd.read_csv('./data/record_all_predictions.csv').dropna()
        df['Date'] = pd.to_datetime(df['Date']) 
        if type(self.start) != int :
            df = df[df['Date'] > self.start]
        df['status'] = df['Prediction']*df['Outcome']
        df['status'][df['status']<0] = 0
        record = df.groupby(['Date']).mean().reset_index()
        print('\nMean daily accuracy of all predictions is {}'.format(np.round(100*np.mean(record['status']),2)))
        
        plt.plot(record['Date'],record['status']*100,'r', label='Daily accuracy all')
        
        df = df.groupby(['Date','Traded']).sum()#.reset_index()
        df['Outcome'][df['Outcome']>0] = 1
        df['Outcome'][df['Outcome']<=0] = -1
        df['Prediction'][df['Prediction']>0] = 1
        df['Prediction'][df['Prediction']<=0] = -1
        df['status'] = df['Prediction']*df['Outcome']
        df['status'][df['status']<0] = 0
        record = df.groupby(['Date']).mean().reset_index()
        print('\nMean daily tradable accuracy of all predictions is {}'.format(np.round(100*np.mean(record['status']),2)))
        
        plt.plot(record['Date'],record['status']*100,'g', label='Daily accuracy all tradable')
        plt.xticks(rotation=45, fontsize=8)
        plt.title("Quality of predictions and trades since {}".format(self.start.strftime("%d/%m/%Y")))
        plt.xlabel('Date')
        plt.ylabel('Accuracy')
        plt.axhline(y=50, color='k', linestyle='-',label='Coin flip accuracy')
        for date in self.avoid.iloc[:-1] :
            plt.axvline(x=date, color='y', linestyle='-')
        plt.axvline(x=self.avoid.iloc[-1], color='y', linestyle='-',label='Days avoided')
        plt.legend()
        if self.save :
            plt.savefig('./Images/trade_quality.png')
            plt.close()
        else :
            plt.show()

    def results_traded(self) :
        record = pd.read_csv('./data/record_traded.csv')
        record['Date'] = pd.to_datetime(record['Date'])
        record['Outcome'] = record['Outcome'].replace(0,random.randint(-1,1))
        if type(self.start) != int :
            record = record[record['Date'] > self.start]
        record = record[~record['Date'].isin(self.avoid)]

        record = record.drop(['Prob_distance'], axis=1)
        
        # Define the traget names
        target_names = ['Down Day', 'Up Day']
        
        y_pred = [int(float(i)) for i in record['Prediction'].tolist()]
        y_true = [int(float(i)) for i in record['Outcome'].tolist()]
        
        # Build a classifcation report
        report = classification_report(y_true = y_true, y_pred = y_pred, target_names = target_names, output_dict = True)
        
        # Add it to a data frame, transpose it for readability.
        report_df = pd.DataFrame(report).transpose()
        report_df
        
        rf_matrix = confusion_matrix(y_true, y_pred)
        
        true_negatives = rf_matrix[0][0]
        false_negatives = rf_matrix[1][0]
        true_positives = rf_matrix[1][1]
        false_positives = rf_matrix[0][1]
        
        accuracy = np.round((true_negatives + true_positives) / (true_negatives + true_positives + false_negatives + false_positives),3)
        percision = np.round(true_positives / (true_positives + false_positives),3)
        recall = np.round(true_positives / (true_positives + false_negatives),3)
        specificity = np.round(true_negatives / (true_negatives + false_positives),3)
        
        if self.verbose :
            print('\nFor traded predictions:')
            print('Accuracy: {}'.format(float(accuracy)))
            print('Percision: {}'.format(float(percision)))
            print('Recall: {}'.format(float(recall)))
            print('Specificity: {}'.format(float(specificity)))
        
        cm = confusion_matrix(y_true, y_pred, normalize='all')
        cmp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=target_names)
        fig, ax = plt.subplots(figsize=(10,10))
        cmp.plot(ax=ax)
        plt.title("Confusion Matrix of trades")
        if self.save :
            plt.savefig('./Images/Traded CM.png')
            plt.close()
        else :
            plt.show()
        
        plt.figure()
        probs = record['Probability'].dropna().tolist()
        y_true = np.array([int(float(i)) for i in record['Outcome'].tolist()])
        y_pred = np.array([int(float(i)) for i in record['Prediction'].tolist()])
        
        probability_pos = []
        probability_neg = []
        probability = []
        for prob in probs :
            probability.append(float(prob))
            if float(prob) > 0.5 :
                probability_pos.append(float(prob))
                     
            else:
                probability_neg.append(1-float(prob))
        
        fpr, tpr, thresholds = metrics.roc_curve(y_true, probability)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='lightgbm')
        display.plot()  
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance', alpha=.8)
        plt.title("ROC traded")
        plt.legend()
        if self.save :
            plt.savefig('./Images/Traded ROC.png')
            plt.close()
        else : 
            plt.show() 
        
        plt.figure()
        bins = np.linspace(0,1,10)
        deltas = record['Delta'].dropna()
        df = pd.DataFrame({'d':deltas,'p':probs})
        group = df.groupby(pd.cut(df.p, bins))
        plot_centers = (bins [:-1] + bins [1:])/2
        plot_values = group.d.mean()
        plt.plot(plot_centers, plot_values)
        plt.title("Probability to delta curve traded (averaged by bin)")
        plt.xlabel('Probability')
        plt.ylabel('Delta')
        if self.save :
            plt.savefig('./Images/Traded probability to delta curve.png')
            plt.close()
        else :
            plt.show()
        
        probability = np.array(probability)
        plt.rcParams["figure.figsize"] = 10, 5
        fig,ax = plt.subplots()
        bins = ax.hist(probability, bins=20, range=(0.0, 1), label = 'Probability distribution')
        ax.set_xlabel("Probability threshold",fontsize=14)
        ax.set_ylabel("Count",color="blue",fontsize=14) 
        shape = np.array(bins[1])
        num = np.array(bins[0])
        
        res = y_true/y_pred
        accuracy = []
        centered = []
        for i in range(len(shape)-1) :
            counter = 0
            total = 0
            centered.append((shape[i+1] + shape[i])/2)
            if centered[-1] < 0.5 :
                sub = res[np.where(probability <= centered[-1])[0]]
                right = len(np.where(sub>0)[0])
                total = len(sub)
            else :
                sub = res[np.where(probability >= centered[-1])[0]]
                right = len(np.where(sub>0)[0])
                total = len(sub)
            
            if total > 0 :
                accuracy.append(right/total)
            elif (total == 0) & (i > 0) :
                accuracy.append(accuracy[-1])
            else :
                accuracy.append(1)
        
        accuracy = np.array(accuracy) * 100

        ax2=ax.twinx()
        ax2.plot(centered,accuracy,color="red", label = 'Accuracy above threshold')
        ax2.set_ylabel("Accuracy",color="red",fontsize=14)
        plt.title("Veracity Traded")
        plt.legend()
        if self.save :
            plt.savefig('./Images/Traded veracity.png')
            plt.close()
        else :
            plt.show()
        
    def results_predicted(self) :
        record = pd.read_csv('./data/record_all_predictions.csv')
        record['Date'] = pd.to_datetime(record['Date'])
        record['Outcome'] = record['Outcome'].replace(0,random.randint(-1,1))
        print(len(record))
        if type(self.start) != int :
            record = record[record['Date'] > self.start]
        record['Date'] = pd.to_datetime(record['Date'])
        record = record[~record['Date'].isin(self.avoid)]
        print(len(record))
        
        # Define the traget names
        target_names = ['Down Day', 'Up Day']
        
        y_pred = [int(float(i)) for i in record['Prediction'].tolist()]
        y_true = [int(float(i)) for i in record['Outcome'].tolist()]
        
        # Build a classifcation report
        report = classification_report(y_true = y_true, y_pred = y_pred, target_names = target_names, output_dict = True)
        
        # Add it to a data frame, transpose it for readability.
        report_df = pd.DataFrame(report).transpose()
        report_df
        
        rf_matrix = confusion_matrix(y_true, y_pred)
        
        true_negatives = rf_matrix[0][0]
        false_negatives = rf_matrix[1][0]
        true_positives = rf_matrix[1][1]
        false_positives = rf_matrix[0][1]
        
        accuracy = np.round((true_negatives + true_positives) / (true_negatives + true_positives + false_negatives + false_positives),3)
        percision = np.round(true_positives / (true_positives + false_positives),3)
        recall = np.round(true_positives / (true_positives + false_negatives),3)
        specificity = np.round(true_negatives / (true_negatives + false_positives),3)
        
        if self.verbose :
            print('\nFor all predictions:')
            print('Accuracy: {}'.format(float(accuracy)))
            print('Percision: {}'.format(float(percision)))
            print('Recall: {}'.format(float(recall)))
            print('Specificity: {}'.format(float(specificity)))
        
        cm = confusion_matrix(y_true, y_pred, normalize='all')
        cmp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=target_names)
        fig, ax = plt.subplots(figsize=(10,10))
        cmp.plot(ax=ax)
        plt.title("Confusion Matrix of predictions")
        if self.save :
            plt.savefig('./Images/Predicted CM.png')
            plt.close()
        else :
            plt.show()
        
        probs = record['Probability'].dropna().tolist()
        y_true = np.array([int(float(i)) for i in record['Outcome'].tolist()])
        y_pred = np.array([int(float(i)) for i in record['Prediction'].tolist()])
        
        probability_pos = []
        probability_neg = []
        probability = []
        for prob in probs :
            probability.append(float(prob))
            if float(prob) > 0.5 :
                probability_pos.append(float(prob))
                     
            else:
                probability_neg.append(1-float(prob))
        
        fpr, tpr, thresholds = metrics.roc_curve(y_true, probability)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='lightgbm')
        display.plot()  
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance', alpha=.8)
        plt.title("ROC all predictions")
        plt.legend()
        if self.save :
            plt.ioff()
            plt.savefig('./Images/Predicted ROC.png')
            plt.close()
        else :
            plt.show() 
        
        plt.figure()
        bins = np.linspace(0,1,20)
        deltas = record['Delta'].dropna().tolist()
        df = pd.DataFrame({'d':deltas,'p':probs})
        group = df.groupby(pd.cut(df.p, bins))
        plot_centers = (bins [:-1] + bins [1:])/2
        plot_values = group.d.mean()
        plt.plot(plot_centers, plot_values)
        plt.title("Probability to delta curve all predictions (averaged by bin)")
        plt.xlabel('Probability')
        plt.ylabel('Delta')
        if self.save :
            plt.ioff()
            plt.savefig('./Images/Predicted probability to delta curve.png')
            plt.close()
        else :
            plt.show()
        
        probability = np.array(probability)
        plt.rcParams["figure.figsize"] = 10, 5
        fig,ax = plt.subplots()
        bins = ax.hist(probability, bins=20, range=(0.0, 1), label = 'Probability distribution')
        ax.set_xlabel("Probability threshold",fontsize=14)
        ax.set_ylabel("Count",color="blue",fontsize=14) 
        shape = np.array(bins[1])
        num = np.array(bins[0])
        
        res = y_true/y_pred
        accuracy = []
        centered = []
        for i in range(len(shape)-1) :
            counter = 0
            total = 0
            centered.append((shape[i+1] + shape[i])/2)
            if centered[-1] < 0.5 :
                sub = res[np.where(probability <= centered[-1])[0]]
                right = len(np.where(sub>0)[0])
                total = len(sub)
            else :
                sub = res[np.where(probability >= centered[-1])[0]]
                right = len(np.where(sub>0)[0])
                total = len(sub)
            
            if total > 0 :
                accuracy.append(right/total)
            elif (total == 0) & (i > 0) :
                accuracy.append(accuracy[-1])
            else :
                accuracy.append(1)
        
        accuracy = np.array(accuracy) * 100

        ax2=ax.twinx()
        ax2.plot(centered,accuracy,color="red", label = 'Accuracy above threshold')
        ax2.set_ylabel("Accuracy",color="red",fontsize=14)
        plt.title("Veracity all predictions")
        plt.legend()
        if self.save :
            plt.savefig('./Images/Predicted veracity.png')
            plt.close()
        else :
            plt.show() 
            
    def history(self) :
        start = self.start
        possibilities =  ['INTC', 'AMZN', 'FB', 'AAPL', 'DIS', 'TSLA', 'GOOG', 'GOOGL', 
                               'MSFT', 'NFLX', 'NVDA', 'TWTR', 'AMD', 'WMT', 'JPM', 'SPY', 'QQQ', 'BAC', 'PG']
        
        record_all = pd.read_csv('./data/record_all_predictions.csv')
        record_traded = pd.read_csv('./data/record_traded.csv')        
        
        if type(start) != int :
            record_all['Date'] = pd.to_datetime(record_all['Date'])
            record_traded['Date'] = pd.to_datetime(record_traded['Date'])
            record_all = record_all[record_all['Date']>start]
            record_traded = record_traded[record_traded['Date']>start]
        
        record_traded = record_traded[~record_traded ['Date'].isin(self.avoid)]
        record_all = record_all[~record_all['Date'].isin(self.avoid)]
        
        df_right = record_all[record_all['Prediction'] == record_all['Outcome']]
        right = []
        for p in possibilities :
            right.append(len(df_right[df_right['Traded']==p]))
            
        df_wrong = record_all[record_all['Prediction'] != record_all['Outcome']]
        wrong = []
        for p in possibilities :
            wrong.append(len(df_wrong[df_wrong['Traded']==p]))
        
        accuracy = np.round(100*sum(right)/(sum(right)+sum(wrong)),2)
        x_pos = [i for i, _ in enumerate(right)]
        
        stock_accuracy = 100*np.array(right)/(np.array(right)+np.array(wrong))
        
        plt.figure()
        plt.bar(x_pos, stock_accuracy, label='stock accuracy')
        plt.axhline(y=50, color='k', linestyle='-',label='Coin flip accuracy')
        plt.ylabel("Accuracy")
        plt.title("{} Predictions made since {}\nAccuracy is {}%".format(len(record_all), start.strftime("%d/%m/%Y"),accuracy))
        plt.xticks(x_pos, possibilities, rotation = 45, fontsize=8)
        plt.legend()
        
        if self.save :
            plt.savefig('./Images/History predicted.png')
            plt.close()
        else :
            plt.show()
        

        df_right = record_traded[record_traded['Prediction'] == record_traded['Outcome']]
        right = []
        for p in possibilities :
            right.append(len(df_right[df_right['Traded']==p]))
            
        df_wrong = record_traded[record_traded['Prediction'] != record_traded['Outcome']]
        wrong = []
        for p in possibilities :
            wrong.append(len(df_wrong[df_wrong['Traded']==p]))
        
        accuracy = np.round(100*sum(right)/(sum(right)+sum(wrong)),2)
        x_pos = [i for i, _ in enumerate(right)]
        
        stock_accuracy = 100*np.array(right)/(np.array(right)+np.array(wrong))
        
        plt.figure()
        plt.bar(x_pos, stock_accuracy, label='stock accuracy')
        plt.axhline(y=50, color='k', linestyle='-',label='Coin flip accuracy')
        plt.ylabel("Accuracy")
        plt.title("{} Trades completed since {}\nAccuracy is {}%".format(len(record_traded), start.strftime("%d/%m/%Y"),accuracy))
        plt.xticks(x_pos, possibilities, rotation = 45, fontsize=8)
        plt.legend()
        
        if self.save :
            plt.savefig('./Images/History traded.png')
            plt.close()
        else :
            plt.show()
            
    def model_count(self):
        record = pd.read_csv('./data/record_model.csv')
        record['date'] = pd.to_datetime(record['date'])
        
        #record = record.drop_duplicates(subset=['model_name', 'parameters'], keep='first')
        if type(self.start) != int :
            record = record[record['date'] > self.start]
        record['metric'] = (record['accuracy_live'] + record['accuracy_test'])/2
        n_day = record.groupby(['date']).count()['model_name']
        n_pass = record[record['metric']>=60].groupby(['date']).count()['model_name']
        ratio = round(100*n_pass/n_day,2).reset_index()
        plt.plot(ratio['date'],ratio['model_name'],'k')
        plt.xticks(rotation=45, fontsize=8)
        plt.title("Percent of models with accuracy '(live+test)/2' above 60% since {} (retests counted)".format(self.start.strftime("%d/%m/%Y")))
        plt.xlabel('Date')
        plt.ylabel('(Models above 60%)/(total number of models) (in %)')
        if self.save :
            plt.savefig('./Images/model count.png')
            plt.close()
        else :
            plt.show()
            
    def parameter_evolution(self):
        
        def length(x):
            return len(eval(x))
        start = self.start
        #start = datetime(2021, 10, 1, 0, 0, 0, 0)
        record = pd.read_csv('./data/record_model.csv').dropna()
        #record = record.drop_duplicates(subset=['model_name', 'parameters'], keep='first')
        record['length'] = record['used'].apply(length)
        length = random.randint(1,4)
        record = record[record['length']==length]
        record['date'] = pd.to_datetime(record['date'])
        if type(start) != int :
            record = record[record['date'] > start]
        parameter = random.choice(['shaps', 'leaves', 'depth', 'rate', 'bins', 'balance'])
        plt.figure()
        stocks = list(set(record['stock']))
        for stock in stocks :
            record_ = record[record['stock'] == stock]
                
            parameters = record_.parameters.to_list()
            models = record_['model_name'].tolist()
            date = record_.date.to_list()
            shaps = [float(model.split('-')[-1]) for model in models]
            leaves = [int(eval(x)[0][1]) for x in parameters]
            depth = [int(eval(x)[2][1]) for x in parameters]
            rate = [eval(x)[3][1] for x in parameters]
            bins = [int(eval(x)[4][1]) for x in parameters]
            balance = [int(eval(x)[7][1]) for x in parameters]
            
            df = pd.DataFrame({'date':date,'shaps':shaps,'leaves':leaves,'depth':depth,'rate':rate,'bins':bins,'balance':balance})
            df['date'] = pd.to_datetime(df['date'])
            df = df.groupby('date').std().reset_index()
    
            plt.plot(df['date'],df[parameter],label=stock)
            plt.xticks(rotation=45, fontsize=8)
            plt.title("Evolution of std of {} at train length of {}".format(parameter, length))
            plt.xlabel('Date')
            plt.ylabel(str(parameter))
            plt.legend(loc=(1.04,0))
        
        plt.show()
        
    def annualized_gain(self):
            
        account = pd.read_csv('./data/account.csv').dropna()
        account['date'] = pd.to_datetime(account['Date'])
        
        if type(self.start) != int :
            account = account[account['date'] > self.start]
        account['Change_account_%'][account['date'].isin(self.avoid)] = 0
        n_days = len(account)
        r_days_traded = (n_days - len(self.avoid))/n_days

        if self.verbose :
            print('\nAverage percent of trading days traded {}%'.format(np.round(100*r_days_traded,1)))
        temp = []
        for i in range(1, len(account['Change_account_%'])+1) :
            average_change = np.sum(account['Change_account_%'].iloc[:i])/i
            temp.append(np.round(((1+average_change/100)**(253) - 1) * 100,2))
            
        account['annual'] = temp
        plt.figure()
        plt.plot(account['Date'],account['annual'],'k', label='Daily annualized gain')
        plt.xticks(rotation='vertical')
        plt.xlabel('Date')
        plt.ylabel('Annualised gain (%)')
        plt.axhline(y=0, color='r', linestyle='-',label='Break even')
        plt.legend()
        plt.xticks(rotation=45, fontsize=8)
        annualised_gain = account['annual'].iloc[-1]
        plt.title("Compacted annualised gain since {}\n Annualised gain taking into account variations before and including given date\n Lastest annualized gain {} %"\
                  .format(self.start.strftime("%d/%m/%Y"), annualised_gain))
        if self.save :
            plt.savefig('./Images/annualized gain.png')
            plt.close()
        else :
            plt.show()
        
    def prediction_control (self):
        def gain(changes,factor):
            value = [100000]
            for ch, fa in zip(changes,factor):
                value.append(value[-1]*(1+ch*(1/fa)/100))
            return value[1:]
    
        account = pd.read_csv('./data/account.csv').dropna()
        preds = pd.read_csv('./data/record_all_predictions.csv')
        account['Date'] = pd.to_datetime(account['Date'])
        preds['Date'] = pd.to_datetime(preds['Date'])
        if type(self.start) != int :
            account = account[account['Date'] > self.start]
            preds  = preds[preds['Date'] > self.start]
        account['outcome'] = account['Change_account_%']
        account.loc[account['outcome']>0,'outcome'] = 1
        account.loc[account['outcome']<=0,'outcome'] = -1
        accuracy = 100*(account[account['outcome']>0].count()/account['outcome'].count()).outcome
        print('\nAccount variation accuracy is {}%'.format(np.round(accuracy,3)))
        account = account.set_index(['Date'])
        account = account.join(preds.groupby(['Date']).sum().abs()['Prediction']/preds.groupby(['Date']).count()['Prediction'], how='inner')
        account = account.reset_index()
        
        levels = np.linspace(0,1,200)
        result = []
        level = []
        for i in levels :
            test2 = account[account['Prediction']<i]
            if len(test2)>0 :
                result.append(100*(gain(test2['Change_account_%'],test2['Trade_factor'])[-1]-100000)/100000)
                level.append(i)
        plt.figure()  
        plt.plot(level,result,label='Account variation at given value of direction level')
        plt.xlabel('Direction level')
        plt.ylabel('Account variation %')
        plt.title ('Maximum found for {}'.format(np.round(level[np.argmax(result)],3)))
        plt.legend()
        if self.save :
            plt.savefig('./Images/Prediction direction.png')
            plt.close()
        else :
            plt.show()
                  
if __name__ == "__main__":
    main()
