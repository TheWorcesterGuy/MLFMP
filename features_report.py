#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 23:07:59 2021

@author: christian
"""

import pandas as pd
import numpy as np


def features_report() :
    features = pd.read_csv('./data/model_features.csv')
    price_data = pd.read_csv('./data/features_store.csv',',')
    features_present_top = []
    features_present_bottom = []
    
    max_line = 10
    
    #Top
    for ii in range(0,max_line) :
        features_present_top += features.iloc[ii].tolist()
    
    features_present_top = list(set(features_present_top))
    df = pd.DataFrame({'Features':features_present_top})
    
    for ii in range(0,max_line) :
        percentage_top = []
        line = features.iloc[ii].tolist()
        for feature in features_present_top:
            percentage_top.append(np.round(100*line.count(feature)/len(line),1))
        
        df['Top '+str(ii+1)] = percentage_top
        
    df = df.reset_index(drop=True)
    df = df.set_index(['Features'])
    Top = df.sort_values(by='Top 1', ascending=False)
    print(Top.head(20))
    Top.head(20).to_csv('./data/top.csv')
    
    #Bottom
    features_rotated = pd.DataFrame()
    for column in features :
        line = features[column].dropna().tolist()
        line.reverse()
        features_rotated[column] = line[:max_line]
        
    for ii in range(0,max_line) :
        features_present_bottom += features_rotated.iloc[ii].tolist()
    
    features_present_bottom = list(set(features_present_bottom))
    df = pd.DataFrame({'Features':features_present_bottom})
    
    for ii in range(0,max_line) :
        percentage_bottom = []
        line = features_rotated.iloc[ii].tolist()
        for feature in features_present_bottom:
            percentage_bottom.append(np.round(100*line.count(feature)/len(line),1))
        
        df['Bottom '+str(ii+1)] = percentage_bottom
        
    df = df.reset_index(drop=True)
    df = df.set_index(['Features'])
    Bottom = df.sort_values(by='Bottom 1', ascending=True)
    print(Bottom.tail(20))  
    Bottom.tail(20).to_csv('./data/Bottom.csv')  
    
    
    #Not used
    all_features_used = []
    for column in features.columns :
            all_features_used += features[column].tolist()
    
    all_features_used = list(set(all_features_used))
    all_features = price_data.columns
    features_not_used = list(set(all_features) - set(all_features_used))
    
    features_not_used = pd.DataFrame({'Features Not Used':features_not_used})
    features_not_used.to_csv('./data/features_not_used.csv', index=False)
    
def top_50():
    features = pd.read_csv('./data/model_features.csv')
    price_data = pd.read_csv('./data/features_store.csv',',')
    features_present_top = []
    features_present_bottom = []
    
    max_line = 50
    
    #Top
    for ii in range(0,max_line) :
        features_present_top += features.iloc[ii].tolist()
    
    features_present_top = list(set(features_present_top))
    df = pd.DataFrame({'Features':features_present_top})
    
    for ii in range(0,max_line) :
        percentage_top = []
        line = features.iloc[ii].tolist()
        for feature in features_present_top:
            percentage_top.append(np.round(100*line.count(feature)/len(line),1))
        
        df['Top '+str(ii+1)] = percentage_top
        
    df = df.reset_index(drop=True)
    df = df.set_index(['Features'])
    Top = df.sort_values(by='Top 1', ascending=False)
    print(Top.head(50))
    Top.head(50).to_csv('./data/top_50.csv')
    
