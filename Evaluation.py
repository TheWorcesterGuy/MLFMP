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
    #Status takes : 'all', 'Archived', 'Deleted', 'Selected', 'xall'
    evaluation(Accuracy = 0, AP_score = 0, ROC_AUC = 0, Model_performance = 0, status = 'all', stock = 'all', Duplicates = False).charts()
    evaluation(Accuracy = 0, AP_score = 0, ROC_AUC = 0, Model_performance = 0, status = 'all', stock = 'all', Duplicates = False).time()
    evaluation(Accuracy = 0, AP_score = 0, ROC_AUC = 0, Model_performance = 0, status = 'all', stock = 'all', Duplicates = True).results_traded()
    evaluation(Accuracy = 0, AP_score = 0, ROC_AUC = 0, Model_performance = 0, status = 'all', stock = 'all', Duplicates = True).results_predicted()
    evaluation(Accuracy = 0, AP_score = 0, ROC_AUC = 0, Model_performance = 0, status = 'all', stock = 'all', Duplicates = True).optimizer()
class evaluation :
    def __init__(self, Accuracy, AP_score, ROC_AUC, Model_performance, status, stock, Duplicates):
        self.accuracy = Accuracy
        self.AP = AP_score
        self.ROC_AUC = ROC_AUC
        self.status = status
        self.Model_performance = Model_performance
        self.product = stock
        self.Duplicates = Duplicates
        record = pd.read_csv('./data/record_model.csv').dropna()
        record['Date'] = pd.to_datetime(record['Date'])
        end_date = datetime.today() - pd.Timedelta("1 days")
        record = record.loc[(record['Date'] > end_date)]
        record = record[record['Long_AP_score'] > self.AP]
        record = record[record['Long_ROC_AUC'] > self.ROC_AUC]
        record = record[record['Long_accuracy'] > self.accuracy]
        record = record[record['Long_Model_performance'] > self.Model_performance]
            
        if self.Duplicates == False :
            record = record.drop_duplicates(subset=['Model_name'], keep='last')
        
        self.record = record
        
    def charts (self):
        
        record = self.record
        
        models = np.array(record['Model_name'].tolist())
        
        Long_AP = np.array(record['Long_AP_score'].tolist())
        Long_ROC = np.array(record['Long_ROC_AUC'].tolist())
        Long_accuracy = np.array(record['Long_accuracy'].tolist())
        Long_performance = np.array(record['Long_Model_performance'].tolist())
        Days_Traded_long = np.array(record['Days_traded_long'].tolist())
        
        Weekly_AP = np.array(record['Weekly_AP_score'].tolist())
        Weekly_ROC = np.array(record['Weekly_ROC_AUC'].tolist())
        Weekly_accuracy = np.array(record['Weekly_accuracy'].tolist())
        
        Trade_accuracy = np.array(record['Trade_accuracy'].tolist())
        Weekly_Trade_accuracy = np.array(record['Weekly_Trade_accuracy'].tolist())
        Weekly_Days_traded = np.array(record['Weekly_Days_traded'].tolist())
        
        strong = len(np.where(Weekly_accuracy  > 60)[0])
        
        print('\nNumber of new models,', len(models))
        print ('There are %d stong models, ' % strong)
        print('Representing, ', np.round(100*strong/len(Weekly_accuracy),3), '% of models' )
        print('Average Weekly accuracy is, ', np.round(np.mean(Weekly_accuracy),3))
        print('Average Weekly ROC is, ', np.round(np.mean(Weekly_ROC),3))
        print('Average Weekly traded accuracy is, ', np.round(np.mean(Weekly_Trade_accuracy ),3))
        print('Average return is (long and no threshold), ', np.round(np.mean(Long_performance),3))
        print('Max Weekly accuracy is, ', np.round(np.max(Weekly_accuracy),3))
        print('Max Weekly ROC is, ', np.round(np.max(Weekly_ROC),3))
        print('Max Weekly traded accuracy is, ', np.round(np.max(Weekly_Trade_accuracy ),3))
        print('Max return is (long and no threshold), ', np.round(np.max(Long_performance),3))
        
        models = np.array(record['Model_name'].tolist())
        
        plt.plot(Long_accuracy, Long_AP,'*')
        plt.title("f(accuracy) = AP_score")
        z = np.polyfit(Long_accuracy.flatten(),  Long_AP.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(Long_accuracy,p(Long_accuracy),"r--", label = "y=%.6fx+%.6f"%(z[0],z[1]))
        plt.legend()
        plt.show()
        
        plt.plot(Weekly_ROC, Weekly_Trade_accuracy,'*')
        plt.title("f(ROC_AUC (weekly)) = Weekly trade accuracy")
        z = np.polyfit(Weekly_ROC.flatten(), Weekly_Trade_accuracy.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(Weekly_ROC, p(Weekly_ROC),"r-", label = "y=%.6fx+%.6f"%(z[0],z[1]))
        plt.legend()
        plt.show()
        
        plt.plot(Long_accuracy, Long_performance,'*')
        plt.title("f(accuracy) = model_performance")
        z = np.polyfit(Long_accuracy.flatten(),  Long_performance.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(Long_accuracy,p(Long_accuracy),"r--", label = "y=%.6fx+%.6f"%(z[0],z[1]))
        plt.legend()
        plt.show()
        
        plt.plot(Long_ROC, Weekly_ROC,'*')
        plt.title("f(ROC) = Weekly_ROC")
        z = np.polyfit(Long_ROC.flatten(),  Weekly_ROC.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(Long_ROC,p(Long_ROC),"r--", label = "y=%.6fx+%.6f"%(z[0],z[1]))
        plt.legend()
        plt.show()
        
        used = record['Used'].tolist()
        
        length = []
        for use in used :
            res = ast.literal_eval(use)
            length.append(len(res))
        
        points = list(set(length))
        length = np.array(length)
        Long_accuracy = np.array(Long_accuracy)
        y_value = []
        y_error = []
        for point in points :
            pos = np.where(length == point)
            y_error.append(np.std(Long_accuracy[pos]))
            y_value.append(np.mean(Long_accuracy[pos]))
            
        plt.errorbar(points, y_value,
             yerr = y_error,
             fmt ='o')
        plt.title("f(length_used) = Accuracy")
        z = np.polyfit(np.array(points).flatten(),  np.array(y_value).flatten(), 1)
        p = np.poly1d(z)
        plt.plot(np.array(points), p(np.array(points)),"r-", label = "y=%.6fx+%.6f"%(z[0],z[1]))
        plt.legend()
        plt.show()
        
        features = []
        for model in models :
            n = float(model.split('-')[1])
            if n == 0:
                n = 100
            
            features.append(n)
            
        points = (np.linspace(30,500,int((500-30)/30))).astype(int)
        features = np.array(features)
        Long_accuracy = np.array(Long_accuracy)
        feature_n = []
        y_value = []
        y_error = []
        for point in points :
            pos = np.where(((point - 30) < features) & (features < point + 30))
            if len(pos[0]) > 0 :
                feature_n.append(np.mean(features[pos]))
                y_error.append(np.std(Long_accuracy[pos]))
                y_value.append(np.mean(Long_accuracy[pos]))
        
        plt.errorbar(feature_n, y_value,
             yerr = y_error,
             fmt ='o')
        plt.title("f(features) = Accuracy")
        z = np.polyfit(np.array(feature_n).flatten(),  np.array(y_value).flatten(), 1)
        p = np.poly1d(z)
        plt.plot(np.array(feature_n), p(np.array(feature_n)),"r-", label = "y=%.6fx+%.6f"%(z[0],z[1]))
        plt.legend()
        plt.show()
        
        # accuracy_no_weight = []
        # accuracy_weight = []
        # counter = 0
        # for model in models :
        #     res = [int(i) for i in model.split('-') if i.isdigit()]
        #     if len(res) > 2 :
        #         if res[1] == 0:
        #             accuracy_no_weight.append(Weekly_Trade_accuracy[counter])
        #         if res[1] == 1:
        #             accuracy_weight.append(Weekly_Trade_accuracy[counter])
        #     counter += 1
                    
        # plt.plot([0, 1],[np.mean(accuracy_no_weight),np.mean(accuracy_weight)], 'x')
        # plt.xticks([0,1])
        # plt.show()
        # print('\nNumber without weights,', len(accuracy_no_weight))
        # print('Number with weights,', len(accuracy_weight))
        
        return
    
    def time(self):
        record = pd.read_csv('./data/record_model.csv')
        record['Date'] = pd.to_datetime(record['Date'])
        mean = record.groupby('Date').mean()
        plt.figure()
        mean['Long_Model_performance'].plot()
        

    def results_traded(self) :
        record = pd.read_csv('./data/record_traded.csv')
        record.replace('wait', np.nan, inplace=True)
        record['Date'] = pd.to_datetime(record['Date'])
        record = record.drop(['Prob_distance'], axis=1)
        
        # Define the traget names
        target_names = ['Down Day', 'Up Day']
        
        y_pred = [int(float(i)) for i in record['predictions'].tolist()]
        y_true = [int(float(i)) for i in record['outcome'].tolist()]
        
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
        y_true = np.array([int(float(i)) for i in record['outcome'].tolist()])
        y_pred = np.array([int(float(i)) for i in record['predictions'].tolist()])
        
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
        
        y_pred = [int(float(i)) for i in record['predictions'].tolist()]
        y_true = [int(float(i)) for i in record['outcome'].tolist()]
        
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
        y_true = np.array([int(float(i)) for i in record['outcome'].tolist()])
        y_pred = np.array([int(float(i)) for i in record['predictions'].tolist()])
        
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