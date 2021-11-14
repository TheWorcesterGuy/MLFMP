#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 03:31:17 2021

@author: christian
"""

import os
import sys
import time
from datetime import datetime, timedelta
import pytz
from email_updates_error import *

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
    while days_running < 100 :
        nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
        if (nyc_datetime.weekday() != 5) and (nyc_datetime.weekday() != 6) :
            print('\n In Week trading \n')
            nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
            end = nyc_datetime.replace(hour=9, minute=30, second=0,microsecond=0)
            start = nyc_datetime.replace(hour=8, minute=35, second=0,microsecond=0)
            if (nyc_datetime < start):
                nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
                intermediate = nyc_datetime.replace(hour=7, minute=21, second=0,microsecond=0)
                difference = intermediate - nyc_datetime
                print('\n Sleeping before google update', round((difference.seconds/60)/60,3), 'hours\n')
                time.sleep(difference.seconds+1)
                print('\n updating features store for google \n')
                os.system("python3 update_features_store.py >> ./log/first" + nyc_datetime.strftime('%Y-%m-%d') + ".txt")
                nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
                start = nyc_datetime.replace(hour=8, minute=35, second=1,microsecond=0)
                difference = start - nyc_datetime
                print('\n Sleeping before market', round((difference.seconds/60)/60,3), 'hours\n')
                time.sleep(difference.seconds+1)
                
            elif (nyc_datetime > end):
                start = (nyc_datetime + timedelta(days=1)).replace(hour=7, minute=21, second=0,microsecond=0)
                nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
                difference = start - nyc_datetime
                print('\n Sleeping', round((difference.seconds/60)/60,3), 'hours\n')
                time.sleep(difference.seconds)
            
            nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
            start = nyc_datetime.replace(hour=8, minute=35, second=0,microsecond=0)
            end = nyc_datetime.replace(hour=9, minute=30, second=0,microsecond=0)
            if (nyc_datetime >= start) & (nyc_datetime < end) :
                os.system("python3 update_features_store.py >> ./log/second" + nyc_datetime.strftime('%Y-%m-%d') + ".txt")
                os.system("python3 trade.py >> ./log/trade_log" + nyc_datetime.strftime('%Y-%m-%d') + ".txt")

            nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
            if (nyc_datetime.weekday() == 4):
                end_week = (nyc_datetime + timedelta(days=1)).replace(hour=1, minute=0, second=1,microsecond=0)
                difference = end_week - nyc_datetime
                print('\n Friday evening stoppage, sleeping', round((difference.seconds/60)/60,3), 'hours\n')
                time.sleep(difference.seconds)
                
        elif (nyc_datetime.weekday() == 5) :
            nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
            end_week = (nyc_datetime + timedelta(days=2)).replace(hour=0, minute=0, second=1,microsecond=0)
            difference = end_week - nyc_datetime
            print('\n Weekend stoppage (Saturday), sleeping', round((difference.seconds/60)/60,3), 'hours\n')
            time.sleep(difference.seconds)
            
        elif (nyc_datetime.weekday() == 6) :
            nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
            end_week = (nyc_datetime + timedelta(days=1)).replace(hour=0, minute=0, second=1,microsecond=0)
            difference = end_week - nyc_datetime
            print('\n Weekend stoppage (Sunday), sleeping', round((difference.seconds/60)/60,3), 'hours\n')
            time.sleep(difference.seconds)
             
        days_running += 1
    

if __name__ == "__main__":
    main()