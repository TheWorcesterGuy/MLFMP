#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 02:44:14 2021

@author: christian
"""

import os
import glob
import pandas as pd

def clean() :
    if len(glob.glob('models')) :
        os.system('rm -r models')
        
    if len(glob.glob('Best_models')) :
        os.system('rm -r Best_models')
        
    if len(glob.glob('./models_in_use')) :
        os.system('rm -r ./models_in_use')
        
    if len(glob.glob('./data/model_report.csv')) :
        os.system('rm ./data/model_report.csv')
        
    if len(glob.glob('./data/Bottom.csv')) :
        os.system('rm ./data/Bottom.csv')
        
    if len(glob.glob('./data/Top.csv')) :
        os.system('rm ./data/Top.csv')
        
    if len(glob.glob('./data/features_not_used.csv')) :
        os.system('rm ./data/features_not_used.csv')
        
    if len(glob.glob('./data/stock_links.csv')) :
        os.system('rm ./data/stock_links.csv')
        
    if len(glob.glob('./data/record_model.csv')) :
        os.system('rm ./data/record_model.csv')
        
    if len(glob.glob('./data/model_features.csv')) :
        os.system('rm ./data/model_features.csv')
        
    if len(glob.glob('./data/history_google_updates.csv')) :
        os.system('rm ./data/history_google_updates.csv')
        
    make()
        
def make() :
    os.system('mkdir ./models')
    os.system('mkdir ./Best_models')
    os.system('mkdir ./models_in_use')
    os.system('touch ./data/history_google_updates.csv')
    
def clean_features() : 
    models = glob.glob('./models/*csv') + glob.glob('./Best_models/*csv')
    models = [x.split('/')[-1].replace('.csv','') for x in models]
    models = list(set(models))
    features = pd.read_csv('./data/model_features.csv')
    to_remove = [x for x in features.columns if x not in models]
    if len(to_remove)>0:
        features = features.drop(to_remove, axis=1)
        features.to_csv('./data/model_features.csv', index = False)
    