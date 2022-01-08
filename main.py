#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 03:31:17 2021

@author: christian
"""

import os
import sys
import time
from datetime import datetime, timedelta, date
import holidays
import pytz
from email_updates_error import *
import glob

def main():
    try:
        trade_system()
    except:
        today = datetime.today()
        error_location = "main.py"
        error_message = "A fatal arror occured in the main trading system organiser,his will stop all trading, caution if this email arrives after todays trades email, then trades need to be manually closed"
        error_report = pd.DataFrame({'Date' : today.strftime('%Y - %m - %d'), 'Code_name' : [error_location], 'Message' : [error_message]})
        error_report = error_report.set_index('Date')
        error(error_report) 
        print(error_message)
        print('Sub-code sleeping until user kill')
        time.sleep(10000000)
    
def trade_system():
    """Function used to execute trading system at correct times. 
    Function evaluates timing based on :
        - Week day or weekend.
        - Friday or other weekday (for weekend stoppage)
    
    Code Calls
    ----------
    `python update_features_store.py` : python code
        Code used to update features store, if this fails all subsequent trading will be blocked"
    `python trade.py` : python code
        Code used to make predictions and subsequently make the trades"
    
    Sub-Code Calls
    --------------
    `python trade.py` calls `python alpaca_trading.py` : python code
        Sub-code used to execute trades"
    `python alpaca_trading.py` calls `email_updates_morning.py` : python code
        Sub_code used to update user of days trades by email"
    `python trade.py` calls `email_updates_evening.py` : python code
        Sub-code used to update user of the outcome of the days trades
    """
    
    print('\nExecuting Wilkinson & Chabannes trading system \n')
    print('Sub codes called in this system can be changed and updated outside of market hours')
    path = os.getcwd()
    days_running = 0
    us_holidays = holidays.UnitedStates()
    us_holidays.append(['2022-11-25']) # Half day
    
    while days_running < 350 :
        
        nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
        if (nyc_datetime.weekday() not in [5,6]) & (nyc_datetime.date() not in us_holidays) :
            print('\n In Week trading \n')
            
            
            nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
            start_first_update = nyc_datetime.replace(hour=7, minute=10, second=2,microsecond=0)
            if (nyc_datetime < start_first_update):
                
                nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
                difference = start_first_update - nyc_datetime
                print('\nSleeping before first update', round((difference.seconds/60)/60,3), 'hours\n')
                time.sleep(difference.seconds+1)
                
                print('\nFirst early morning update of features store \n')
                os.system("python3 update_features_store.py > ./log/features_store/update_log.txt")
                print('\nFirst early morning update of features store --- Completed\n')
                
                nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
                difference = nyc_datetime.replace(hour=8, minute=10, second=2,microsecond=0) - nyc_datetime
                print('\n Sleeping before market', round((difference.seconds/60)/60,3), 'hours\n')
                time.sleep(difference.seconds+1)
            
            
            nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
            start_trade = nyc_datetime.replace(hour=8, minute=10, second=1,microsecond=0)
            end_trade = nyc_datetime.replace(hour=8, minute=50, second=0,microsecond=0)
            if (nyc_datetime < end_trade) :
                
                if (nyc_datetime < start_trade) :
                    difference = start_trade - nyc_datetime
                    print('\n Sleeping before market', round((difference.seconds/60)/60,3), 'hours\n')
                    time.sleep(difference.seconds+1)
                
                print('\nFinal update of features store \n')
                os.system("python3 update_features_store.py > ./log/features_store/update_log.txt")
                print('\nFinal update of features store --- Completed\n')
                print('\nIntiating trading system\n')
                
                if len(glob.glob('./data/features_store.csv')) :
                    os.system("python3 trade.py > ./log/trading/trade_log" + nyc_datetime.strftime('%Y-%m-%d') + ".txt")
                else :
                    print('\n Features store not available, sleeping until user intervention (or new cycle)')
                    time.sleep(43200)
                    
                    
            nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))   
            end_trade = nyc_datetime.replace(hour=8, minute=35, second=0,microsecond=0)
            if (nyc_datetime > end_trade):
                start_first_update = (nyc_datetime + timedelta(days=1)).replace(hour=7, minute=0, second=0,microsecond=0)
                difference = start_first_update - datetime.now(pytz.timezone('US/Eastern'))
                print('\n Out of model trading hours, sleeping :', round((difference.seconds/60)/60,3), 'hours\n')
                time.sleep(difference.seconds)
                
                
        nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))     
        if (nyc_datetime.weekday() == 5) :
            end_week = (nyc_datetime + timedelta(days=1)).replace(hour=0, minute=0, second=1,microsecond=0)
            difference = end_week - nyc_datetime
            print('\n Weekend stoppage (Saturday), sleeping', round((difference.seconds/60)/60,3), 'hours\n')
            time.sleep(difference.seconds)
            os.system("python3 update_features_store.py > ./log/features_store/update_log.txt")
            
            
        nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))   
        if (nyc_datetime.weekday() == 6) :
            end_week = (nyc_datetime + timedelta(days=1)).replace(hour=0, minute=0, second=1,microsecond=0)
            difference = end_week - nyc_datetime
            print('\n Weekend stoppage (Sunday), sleeping', round((difference.seconds/60)/60,3), 'hours\n')
            time.sleep(difference.seconds)
            
            
        nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))   
        if (nyc_datetime.date() in us_holidays) :
            next_day = (nyc_datetime + timedelta(days=1)).replace(hour=0, minute=0, second=1,microsecond=0)
            difference = next_day - nyc_datetime
            print('\n Bank holiday stoppage, sleeping', round((difference.seconds/60)/60,3), 'hours\n')
            time.sleep(difference.seconds)
             
        days_running += 1
    

if __name__ == "__main__":
    main()