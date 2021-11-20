#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import alpaca_trade_api as tradeapi
from datetime import datetime
from alpaca_trade_api.rest import TimeFrame
import time
from datetime import datetime, timedelta
import pytz
import os
import pandas as pd
import numpy as np
import warnings
#from email_updates_error import *
import yfinance as yf
import sys
import time
import finnhub
import glob
import random
from functools import reduce



warnings.simplefilter(action='ignore')



def main():


    stock = sys.argv[1]
    #stock = 'SPY'
    
    df = pd.read_csv('./data/TRADE_DATA/minute_data/%s.csv' % stock)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime')
    
    df.index = df['Datetime']
    
    df_regular = df.between_time('9:30', '16:00')[['Datetime', 'open', 'trade_count']]
    df_extra = df.between_time('16:00', '23:59')[['Datetime', 'open', 'trade_count']]
    
    
    # compute intra days price delta
    df_delta_14_16 = get_high_res_delta(df_regular, '14:00', '16:00')
    df_delta_15_16 = get_high_res_delta(df_regular, '15:00', '16:00')
    df_delta_15h30_16 = get_high_res_delta(df_extra, '16:00', '18:00')
    df_delta_16_23 = get_high_res_delta(df_extra, '16:00', '23:59')
    
    # compute percentage of past-time volume
    df_regular['Date'] = df_regular['Datetime'].dt.date  
    df_extra['Date'] = df_extra['Datetime'].dt.date   
     
    df_regular_trade_count = df_regular[['Date', 'trade_count']].groupby(['Date']).sum()
    df_extra_trade_count = df_extra[['Date', 'trade_count']].groupby(['Date']).sum()
    
    df_regular_trade_count = df_regular_trade_count.rename(columns={'trade_count':'trade_count_regular'})
    df_extra_trade_count = df_extra_trade_count.rename(columns={'trade_count':'trade_count_extra'})
    
    df_trade_count = df_regular_trade_count.merge(df_extra_trade_count, on='Date', how='inner')
    df_trade_count['percentage_volume_extra_time'] = df_trade_count['trade_count_regular'] / (df_trade_count['trade_count_extra'] + df_trade_count['trade_count_extra']) * 100
    df_trade_count = df_trade_count.reset_index(drop=False)
    df_trade_count = df_trade_count[['Date', 'percentage_volume_extra_time']]

    # join two sources
    df_final = reduce(lambda df1, df2: pd.merge(df1, df2, on='Date', how='outer'), [df_trade_count, df_delta_14_16, df_delta_15_16, df_delta_15h30_16, df_delta_16_23])
    df_final = df_final.fillna(-9999)
    
    # add one day since corresponding features are used next day    
    df_final['Date'] = df_final['Date'].shift(-1)
    # last day is logically a nan and therefore we put the current day's date (stock didn't open yet)
    df_final.loc[(df_final.index == df_final.index.max()) & (df_final['Date'].isna()), 'Date'] = datetime.today().strftime('%Y-%m-%d')
    
    df_final.to_csv('./data/%s_minute_price_features.csv' % stock, index=False)
    
    print(df_final.tail(1))
    
    
    


def get_high_res_delta(df, start, end):
	df_end = df.between_time(end, end).drop_duplicates('Datetime')
	df_start = df.between_time(start, start).drop_duplicates('Datetime')
	    
	df_end = df_end.reset_index(drop=True)[['Datetime', 'open']]
	df_start = df_start.reset_index(drop=True)[['Datetime', 'open']]
	    
	    
	df_end = df_end.rename(columns={'open':'price_%s' % end})
	df_start = df_start.rename(columns={'open':'price_%s' % start})
	    
	df_end['Date'] = df_end['Datetime'].dt.date    
	df_start['Date'] = df_start['Datetime'].dt.date
	    
	df_daily_start_end = df_end.merge(df_start, on='Date', how='inner')
	df_daily_start_end['delta_%s_%s' % (start, end)] = (df_daily_start_end['price_%s' % end] - df_daily_start_end['price_%s' % start]) / df_daily_start_end['price_%s' % start] * 100
	df_daily_start_end = df_daily_start_end[['Date', 'delta_%s_%s' % (start, end)]]
	    
	return df_daily_start_end

if __name__ == "__main__":
    main()
