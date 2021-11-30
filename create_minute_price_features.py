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
    df = df.set_index('Datetime')
    print(df)

    
    # Convert to NY time
    eastern = pytz.timezone('US/Eastern')
    df.index = df.index.tz_convert(eastern)
    df = df.sort_values('Datetime')
    
    # compute intra days price delta
    df_delta_14_16 = get_high_res_delta(df[['open', 'trade_count']], '14:00', '16:00')
    df_delta_15_16 = get_high_res_delta(df[['open', 'trade_count']], '15:00', '16:00')
    df_delta_15h30_16 = get_high_res_delta(df[['open', 'trade_count']], '15:30', '16:00')
    
    
    # compute percentage of past-time volume
    
    df_regular = df.between_time('9:30', '16:00')[['open', 'trade_count']]
    df_extra = df.between_time('16:00', '23:00')[['open', 'trade_count']]
    
    df_regular = df_regular.reset_index(drop=False)
    df_extra = df_extra.reset_index(drop=False)
    
    df_regular['Date'] = df_regular['Datetime'].dt.date  
    df_extra['Date'] = df_extra['Datetime'].dt.date   
     
    df_regular_trade_count = df_regular[['Date', 'trade_count']].groupby(['Date']).sum()
    df_extra_trade_count = df_extra[['Date', 'trade_count']].groupby(['Date']).sum()
    
    df_regular_trade_count = df_regular_trade_count.rename(columns={'trade_count':'trade_count_regular'})
    df_extra_trade_count = df_extra_trade_count.rename(columns={'trade_count':'trade_count_extra'})
    
    df_trade_count = df_regular_trade_count.merge(df_extra_trade_count, on='Date', how='inner')
    df_trade_count['percentage_volume_extra_time'] = df_trade_count['trade_count_extra'] / (df_trade_count['trade_count_regular'] + df_trade_count['trade_count_extra']) * 100
    df_trade_count = df_trade_count.reset_index(drop=False)
    df_trade_count = df_trade_count[['Date', 'percentage_volume_extra_time']]
    df_trade_count['Date'] = pd.to_datetime(df_trade_count['Date'])
    
    
    # get delta closing at 16h - last trade recorded
    price_min_time = df_regular.sort_values('Datetime', ascending=True).drop_duplicates(['Date'], keep='last')[['Date', 'open']]
    price_min_time = price_min_time.rename(columns={'open':'price_close'})
    price_max_time = df_extra.sort_values('Datetime', ascending=True).drop_duplicates(['Date'], keep='last')[['Date', 'open']]
    price_max_time = price_max_time.rename(columns={'open':'price_next_open'})
    df_delta_close_next_open = price_min_time.merge(price_max_time, on='Date', how='outer')
    df_delta_close_next_open['delta_post_closing'] = df_delta_close_next_open['price_close'] / df_delta_close_next_open['price_next_open']
    df_delta_close_next_open = df_delta_close_next_open[['Date', 'delta_post_closing']]
    df_delta_close_next_open['Date'] = pd.to_datetime(df_delta_close_next_open['Date'])


    # join two sources
    df_final = reduce(lambda df1, df2: pd.merge(df1, df2, on='Date', how='outer'), [df_trade_count, df_delta_14_16, df_delta_15_16, df_delta_15h30_16, 
    												df_delta_close_next_open])
    df_final = df_final.sort_values('Date')										    
    
    # add one day since corresponding features are used next day    
    df_final['Date'] = df_final['Date'].shift(-1)
    df_final = df_final.reset_index(drop=True)
    # last day is logically a nan and therefore we put the current day's date (stock didn't open yet)
    df_final.loc[(df_final.index == df_final.index.max()) & (df_final['Date'].isna()), 'Date'] = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
    df_final = df_final.fillna(-9999)
    
    df_final['Date'] = pd.to_datetime(df_final['Date'])
    df_final = df_final.sort_values('Date')
    df_final.to_csv('./data/%s_minute_price_features.csv' % stock, index=False)
    
    print(df_final.tail(10))
    

    
    


def get_high_res_delta(df, start, end):

	end_lower = str(int(end[0:2]) - 1) + end[2:]
	
	df_end = df.between_time(end_lower, end).sort_values('Datetime', ascending=False).reset_index(drop=False)
	df_end['Date'] = df_end['Datetime'].dt.date
	df_end = df_end.drop_duplicates('Date', keep='first')
	
	start_lower = str(int(start[0:2]) - 1) + start[2:]
	df_start = df.between_time(start_lower, start).sort_values('Datetime', ascending=True).reset_index(drop=False)
	df_start['Date'] = df_start['Datetime'].dt.date
	df_start = df_start.drop_duplicates('Date', keep='last')
	    
	df_end = df_end[['Date', 'open']]
	df_start = df_start[['Date', 'open']]
	    
	    
	df_end = df_end.rename(columns={'open':'price_%s' % end})
	df_start = df_start.rename(columns={'open':'price_%s' % start})

	df_daily_start_end = df_end.merge(df_start, on='Date', how='inner')
	df_daily_start_end['delta_%s_%s' % (start, end)] = (df_daily_start_end['price_%s' % end] - df_daily_start_end['price_%s' % start]) / df_daily_start_end['price_%s' % start] * 100
	df_daily_start_end = df_daily_start_end[['Date', 'delta_%s_%s' % (start, end)]]
	
	df_daily_start_end['Date'] = pd.to_datetime(df_daily_start_end['Date'])
	
	
	for col in df_daily_start_end.columns:
		df_daily_start_end = df_daily_start_end.rename(columns={col: col.replace(':', '_')})
	

	return df_daily_start_end

if __name__ == "__main__":
    main()
