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

def main():
    print('\n Evaluating recorded models \n')

    evaluation().charts()
    #evaluation().variable()
    #evaluation().money()
    #evaluation().models_selected()
    #evaluation().results_traded()
    #evaluation().results_predicted()
    #evaluation().optimizer()
    
class evaluation :
    def __init__(self):
        record = pd.read_csv('./data/record_model.csv').dropna()
        record = record.drop_duplicates(subset=['model_name'], keep='last')
        record['date'] = pd.to_datetime(record['date'])
        today = datetime.now()
        recent = today - timedelta(days=1)
        record = record[record['date'] > recent]
        self.record = record.dropna()
        
        
    def charts (self):
        
        record = self.record
        percentage_days = 10
        percentage_days = percentage_days/100
        record = record[record['days_traded_test'] > int(150*percentage_days)]
        record = record[record['days_traded_live'] > int(100*percentage_days)]
        
        #record = record[record['status'] > 0]
        accuracy_test = sorted(record['trade_accuracy_test'].to_list())[:-5]
        accuracy_live = record['trade_accuracy_live'].to_list()
        ROC_test = sorted(record['ROC_test'].to_list())[:-5]
        
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
            
        print('mean_test', np.round(np.mean(accuracy_test),2))
        print('mean_live', np.round(np.mean(accuracy_live),2))
        
        plt.title('Mean live accuracy over threshold')
        plt.xlabel('Test accuracy threshold')
        plt.ylabel('Mean live accuracy above threshold')
        plt.plot(accuracy_test, mean_live_thr, 'o')
        plt.show()
        
        plt.title('Mean live accuracy over ROC threshold')
        plt.xlabel('Test ROC threshold')
        plt.ylabel('Mean live accuracy above threshold')
        plt.plot(ROC_test, mean_live_ROC_thr, 'o')
        plt.show()
        
        # plt.title('Standard deviation of performance over threshold')
        # plt.xlabel('Test accuracy threshold')
        # plt.ylabel('standard deviation above threshold')
        # plt.plot(accuracy_test, std_live_thr, 'o')
        # plt.show()
        
        plt.title('Trading results over threshold')
        plt.xlabel('Test accuracy threshold')
        plt.ylabel('Live results over threshold')
        plt.plot(accuracy_test, mean_live_perf, 'o')
        plt.show()
        
        plt.title('Trading results over ROC threshold')
        plt.xlabel('Test ROC threshold')
        plt.ylabel('Live results over threshold')
        plt.plot(ROC_test, mean_live_ROC_perf, 'o')
        plt.show()

    def variable(self):    
        record = pd.read_csv('./data/record_model.csv').dropna()
        accuracy_test = np.array(record['trade_accuracy_test'].to_list())#[:-5]
        accuracy_live = np.array(record['trade_accuracy_live'].to_list())
        ROC_live = np.array(record['ROC_live'].to_list())#[:-5]
        ROC_test = np.array(record['ROC_test'].to_list())
        metric = (accuracy_test + accuracy_live + ROC_live + ROC_test)/4
        #metric = accuracy_live
        
        # plt.figure()
        # models = record['model_name'].tolist()
        # values = [float(x.split('-')[-1]) for x in models]
        # df = pd.DataFrame({'shaps' : values, 'metric': metric})
        # df = df[df['shaps']>0]
        # df = df.groupby(['shaps']).mean().reset_index()
        # plt.plot(np.array(df['shaps']),np.array(df['metric']),'o')
        # plt.xscale('log')
        # plt.xlabel('Shaps level')
        # plt.ylabel('metric')
        # plt.show
        
        plt.figure()
        record = pd.read_csv('./data/record_model.csv').dropna()
        parameters = record.parameters.to_list()
        learning_rate = [eval(x)[3][1] for x in parameters]
        df = pd.DataFrame({'learning rate' : learning_rate, 'metric': metric})
        df = df.groupby(['learning rate']).mean().reset_index()
        plt.plot(np.array(df['learning rate']),np.array(df['metric']),'o')
        plt.xlabel('learning rate')
        plt.xscale('log')
        plt.ylabel('metric')
        plt.show
        
        plt.figure()
        record = pd.read_csv('./data/record_model.csv').dropna()
        parameters = record.parameters.to_list()
        bins = [eval(x)[4][1] for x in parameters]
        df = pd.DataFrame({'bins' : bins, 'metric': metric})
        df = df.groupby(['bins']).mean().reset_index()
        plt.plot(np.array(df['bins']),np.array(df['metric']),'o')
        plt.xlabel('bins')
        plt.ylabel('metric')
        plt.show
        
    def money(self) :
        
        df = pd.read_csv('./data/record_model.csv').dropna()
        plt.figure()
        df = df.groupby(['accuracy_live']).mean().reset_index()
        plt.plot(np.array(df['accuracy_live']),np.array(df['model_performance_live']),'o')
        plt.xlabel('Live accuracy')
        plt.ylabel('Performance')
        plt.show
        
        df = pd.read_csv('./data/record_model.csv').dropna()
        plt.figure()
        df = df.groupby(['ROC_live']).mean().reset_index()
        plt.plot(np.array(df['ROC_live']),np.array(df['model_performance_live']),'o')
        plt.xlabel('Live ROC')
        plt.ylabel('Performance')
        plt.show
        
        df = pd.read_csv('./data/record_model.csv').dropna()
        plt.figure()
        df = df.groupby(['ROC_live']).mean().reset_index()
        plt.plot((np.array(df['ROC_live'])+np.array(df['accuracy_live']))/2,np.array(df['model_performance_live']),'o')
        plt.xlabel('metric')
        plt.ylabel('Performance')
        plt.show
        
    def models_selected(self):
        record = pd.read_csv('./data/record_model.csv').dropna()
        record = record[record['status']==1]
        record = record.groupby(['date']).sum().reset_index()
        record['date'] = pd.to_datetime(record['date']) 
        plt.figure()
        plt.plot(record['date'],record['status'],'o')
        plt.xticks(rotation='vertical')
        plt.xlabel('Date')
        plt.ylabel('Number of good models')
        plt.show()

    def results_traded(self) :
        record = pd.read_csv('./data/record_traded.csv')
        record.replace('wait', np.nan, inplace=True)
        record['Date'] = pd.to_datetime(record['Date'])
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
        
        #self.ROC = average_precision_score(y_test, probs[:, 1])
        print('\nFor traded predictions:')
        print('Accuracy: {}'.format(float(accuracy)))
        print('Percision: {}'.format(float(percision)))
        print('Recall: {}'.format(float(recall)))
        print('Specificity: {}'.format(float(specificity)))
        
        cm = confusion_matrix(y_true, y_pred)
        cmp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=target_names)
        fig, ax = plt.subplots(figsize=(10,10))
        cmp.plot(ax=ax)
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
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Random Forest')
        display.plot()  
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        plt.title("ROC traded")
        plt.legend()
        plt.show() 
        
        deltas = record['Delta'].dropna().tolist()
        plt.plot(probs, deltas, '*')
        plt.title("Probability to delta curve traded")
        plt.xlabel('Probability')
        plt.ylabel('Delta')
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
        plt.show()
        
    def results_predicted(self) :
        record = pd.read_csv('./data/record_all_predictions.csv')
        # record1 = record[record['Probability'] > 0.7]
        # record2 = record[record['Probability'] < 0.3]
        # record = pd.concat([record1,record2])
        record['Date'] = pd.to_datetime(record['Date'])
        record.replace('wait', np.nan, inplace=True)
        
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
        
        #self.ROC = average_precision_score(y_test, probs[:, 1])
        print('\nFor all predictions:')
        print('Accuracy: {}'.format(float(accuracy)))
        print('Percision: {}'.format(float(percision)))
        print('Recall: {}'.format(float(recall)))
        print('Specificity: {}'.format(float(specificity)))
        
        cm = confusion_matrix(y_true, y_pred)
        cmp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=target_names)
        fig, ax = plt.subplots(figsize=(10,10))
        cmp.plot(ax=ax)
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
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Random Forest')
        display.plot()  
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        plt.title("ROC all predictions")
        plt.legend()
        plt.show() 
        
        deltas = record['Delta'].dropna().tolist()
        plt.plot(probs, deltas, '*')
        plt.title("Probability to delta curve all predictions")
        plt.xlabel('Probability')
        plt.ylabel('Delta')
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
        plt.show() 
        
    def optimizer(self) :
        
        record = pd.read_csv('./data/record_model.csv')
        record['Date'] = pd.to_datetime(record['Date'])
        end_date = datetime.today() - pd.Timedelta("1 days")
        record = record.loc[(record['Date'] > end_date)]
        parameters = record['Parameters'].tolist()
        Weekly_Trade_accuracy = record['Weekly_Trade_accuracy'].tolist()
        df_parameters = pd.DataFrame()
        for i in range (len(parameters)):
            df = pd.DataFrame(eval(parameters[i]), columns=['paramater', 'value'])
            df = df.set_index('paramater')
            df = df.T
            df['accuracy'] = Weekly_Trade_accuracy[i]
            df_parameters = df_parameters.append(df, ignore_index=True)
        
        num_leaves = df_parameters['num_leaves'].to_list()        
        points = list(set(num_leaves))
        num_leaves = np.array(num_leaves)
        accuracy = np.array(df_parameters['accuracy'].to_list())
        y_value = []
        y_error = []
        for point in points :
            pos = np.where(num_leaves == point)
            y_error.append(np.std(accuracy[pos]))
            y_value.append(np.mean(accuracy[pos]))
            
        plt.errorbar(points, y_value,
             yerr = y_error,
             fmt ='o')
        plt.title("f(num_leaves) = Accuracy")
        z = np.polyfit(np.array(points).flatten(),  np.array(y_value).flatten(), 1)
        p = np.poly1d(z)
        plt.plot(np.array(points), p(np.array(points)),"r-", label = "y=%.6fx+%.6f"%(z[0],z[1]))
        plt.legend()
        plt.show()
        
        max_depth = df_parameters['max_depth'].to_list()        
        points = list(set(max_depth))
        max_depth = np.array(max_depth)
        accuracy = np.array(df_parameters['accuracy'].to_list())
        y_value = []
        y_error = []
        for point in points :
            pos = np.where(max_depth == point)
            y_error.append(np.std(accuracy[pos]))
            y_value.append(np.mean(accuracy[pos]))
            
        plt.errorbar(points, y_value,
             yerr = y_error,
             fmt ='o')
        plt.title("f(max_depth) = Accuracy")
        z = np.polyfit(np.array(points).flatten(),  np.array(y_value).flatten(), 1)
        p = np.poly1d(z)
        plt.plot(np.array(points), p(np.array(points)),"r-", label = "y=%.6fx+%.6f"%(z[0],z[1]))
        plt.legend()
        plt.show()
        
        learning_rate = df_parameters['learning_rate'].to_list()        
        points = list(set(learning_rate))
        learning_rate = np.array(learning_rate)
        accuracy = np.array(df_parameters['accuracy'].to_list())
        y_value = []
        y_error = []
        for point in points :
            pos = np.where(learning_rate == point)
            y_error.append(np.std(accuracy[pos]))
            y_value.append(np.mean(accuracy[pos]))
            
        plt.errorbar(points, y_value,
             yerr = y_error,
             fmt ='o')
        plt.title("f(learning_rate) = Accuracy")
        z = np.polyfit(np.array(points).flatten(),  np.array(y_value).flatten(), 1)
        p = np.poly1d(z)
        #plt.plot(np.array(points), p(np.array(points)),"r-", label = "y=%.6fx+%.6f"%(z[0],z[1]))
        plt.legend()
        plt.xscale('log')
        plt.show()
        
        bins = df_parameters['max_bin'].to_list()        
        points = list(set(bins))
        bins = np.array(bins)
        accuracy = np.array(df_parameters['accuracy'].to_list())
        y_value = []
        y_error = []
        for point in points :
            pos = np.where(bins == point)
            y_error.append(np.std(accuracy[pos]))
            y_value.append(np.mean(accuracy[pos]))
            
        plt.errorbar(points, y_value,
             yerr = y_error,
             fmt ='o')
        plt.title("f(bins) = Accuracy")
        z = np.polyfit(np.array(points).flatten(),  np.array(y_value).flatten(), 1)
        p = np.poly1d(z)
        plt.plot(np.array(points), p(np.array(points)),"r-", label = "y=%.6fx+%.6f"%(z[0],z[1]))
        plt.legend()
        plt.show()
        
        
if __name__ == "__main__":
    main()