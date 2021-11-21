#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:05:34 2021

@author: christian
"""

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
from email_updates_error import *
warnings.simplefilter(action = 'ignore')

def main():
    nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
    api, buying_power, account_value = login()
    trade = pd.read_csv('./data/to_trade.csv')
    trade_data, trade_value = distribution(api, buying_power, trade)
    os.system('python email_updates_morning.py')
    open_trade(api, trade_data)
    nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
    close = nyc_datetime.replace(hour=15, minute=59, second=30,microsecond=0)
    difference = close - nyc_datetime
    print('Sleeping : ', difference.seconds )
    time.sleep(difference.seconds)
    close_trades(api, account_value, trade_value)
            

def error_handling(error_message) :
        """Function used to handle errors, will send email using the imported error function from 'email_updates_error'

        Parameters
        ----------
        `error_message` : Information about the error
        """
        
        today = datetime.today()
        error_location = "alpaca_trading.py"
        error_report = pd.DataFrame({'Date' : today.strftime('%Y - %m - %d'), 'Code_name' : [error_location], 'Message' : [error_message]})
        error_report = error_report.set_index('Date')
        error(error_report) 
        print(error_message)
        print('Sub-code sleeping until user kill')
        time.sleep(10000000)
        
def login() :  
    """Function used to verify login to alpaca account, WARNING an error will withheld all possible trades'

    Returned
    --------
    `api` : alpaca_trade_api.rest.REST
        Connection data
    `buying_power` : float
        Available funds to trade
    """
    
    APCA_API_KEY_ID = 'PK6OYPKSRKOJ3EQYT8X6'
    APCA_API_SECRET_KEY = 'c0ij7l8e7uS1ULMPMk6DQMAlwlWKSTrEEfjgHDxq'
    APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
    APCA_API_DATA_URL = 'https://data.alpaca.markets'
    APCA_RETRY_MAX = 3	
    APCA_RETRY_WAIT = 3	
    APCA_RETRY_CODES = 429.504	
    api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY,APCA_API_BASE_URL)
    account = api.get_account()
    
    cash = account.cash 
    account_value = pd.DataFrame({'Date' : datetime.now().strftime('%Y - %m - %d'), 'AM' : [cash]})
    account_value.to_csv('./data/account_value.csv', index = False)
    print('${} is available as buying power.'.format(account.buying_power))
    
    return api, float(account.buying_power), account_value

def distribution(api, buying_power, trade) :
    """Class function used to distribute a reduced buying power between stocks based on probabilities and number of predictions'

    Returned
    --------
    `df` : Pandas DataFrame
        Contains the final trading data, note this data is also exported to '.csv' for email updates
    """
    
    reduced_power = (buying_power/2)
    print(reduced_power)
    stocks = trade['Products'].tolist()
    stocks = list(set(stocks))
    
    Last_Stock_Value = []
    Fractionableionable = []
    side = []
    probability = []
    df = pd.DataFrame(columns = ['product', 'side', 'Value_to_trade_$', 'Fractionable','Last_Stock_Value_$'])
    for stock in stocks :
        barset = api.get_barset(stock, 'day', limit=5)
        bars = barset[stock]
        Last_Stock_Value.append(bars[-1].c + bars[-1].c * 0.02)
        side.append(np.sum(trade['Side'][trade['Products'] == stock].tolist()))
        
        if side[-1] <= 0 :
            frac = False
        else :
            frac = api.get_asset(stock).fractionable
        
        Fractionableionable.append(frac)
        probability.append(np.mean(trade['Prob_distance'][trade['Products']==stock].tolist()))
         
    df['product'] = stocks
    df['side'] = side
    df['Value_to_trade_$'] = probability
    df['Fractionable'] = Fractionableionable
    df['Last_Stock_Value_$'] = Last_Stock_Value
    
    df = df[df['Value_to_trade_$'] != 0]
    df['Value_to_trade_$'] = np.abs(np.array(df['Value_to_trade_$'].tolist()) *  np.array(df['side'].tolist()))
    print(df)
    
    df = df.sort_values(by=['Value_to_trade_$'], ascending=False)
    df['Value_to_trade_$'] = (df['Value_to_trade_$'])/(df['Value_to_trade_$'].sum()) * reduced_power
    df['Quantity'] = df['Value_to_trade_$']/df['Last_Stock_Value_$']
    df['Quantity'][df['Fractionable'] == False] = np.floor(df['Quantity'][df['Fractionable'] == False].tolist())
    df['Quantity'][df['Fractionable'] == True] = np.around(df['Quantity'][df['Fractionable'] == True].tolist(),2)
    df = df[df['Quantity'] > 0]
    trade_value = df['Value_to_trade_$'].sum()
    
    print(df)
    df.to_csv('./data/days_trades.csv', index = False)
    
    return df, trade_value    

def open_trade(api, trade_data) :
    """Function used to open trades based on the stock, trade type and quantity'

    """
    
    positions = api.list_positions()
    orders = api.list_orders(status='open')
    
    print('Postions : If the following detail is not empty then, trade warning')
    print(positions)
    
    print('Orders : If the following detail does not change then, trade warning')
    print(orders)
    
    stocks = trade_data['product'].tolist()
    side = trade_data['side'].tolist()
    Quantity = trade_data['Quantity'].tolist()
    
    for i in range(len(stocks)) :
    
        if side[i] >= 1 :
            api.submit_order(
                symbol = stocks[i],
                qty = Quantity[i],  
                side = 'buy',
                type = 'market',
                time_in_force = 'day',
            )
            
        if side[i] <= -1 :
            api.submit_order(
                symbol = stocks[i],
                qty = Quantity[i],  
                side = 'sell',
                type = 'market',
                time_in_force = 'day',
        )
            
    orders = api.list_orders()
    print('Todays orders sent are :')
    print(orders)

def close_trades(api, account_value, trade_value):
    """Function used to close trades and orders, trades and orders are closed in a loop'
        WARNING : It is essential that this functions is run if trades have been opened, otherwise losses can occure.
    """
    
    orders = api.list_orders(status='open')
    positions = api.list_positions()
    retry = 1
    while (len(orders) + len(positions)) > 0:
        try :
            if orders or positions:
                if positions:
                    print(positions)
            
                if orders:
                    print("Canceling open orders:")
                    print([o.id for o in orders])
                    result = [api.cancel_order(o.id) for o in orders]
                    print(result)
            
                closed = []
                for p in positions:
                    side = 'sell'
                    if float(p.qty) < 0:
                        print(p.qty)
                        side = 'buy'
                    closed.append(
                        api.submit_order(p.symbol, qty=abs(float(p.qty)), side=side, type="market", time_in_force="day")
                        )
            
                if closed:
                    print("Submitted Orders", closed)
            
                for o in closed:
                    status = api.get_order(o.id)
                    if status.status == 'rejected':
                        print("ORDER FAILED: Your Order was Rejected!!!")
            
            time.sleep(2)
            orders = api.list_orders(status='open')
            positions = api.list_positions()
            
        except :
            time.sleep(2)
            print('Error in closing positions, retry :', retry)
            orders = api.list_orders(status='open')
            positions = api.list_positions()
            retry += 1
            
            if retry == 10:
                print('Max retries, sleeping until user intervention')
                time.sleep(1000000)
                
    time.sleep(60)    
    account = api.get_account()
    cash = account.cash
    account_value = pd.read_csv('./data/account_value.csv')
    df = pd.DataFrame({'Date' : datetime.now().strftime('%Y - %m - %d'), 'PM' : [cash], 'Trade_value' : [trade_value]})
    account_value = account_value.merge(df, on='Date')
    account_value['Change_account_%'] = 100*(float(account_value ['PM'].iloc[0]) - float(account_value ['AM'].iloc[0]))/float(account_value ['AM'].iloc[0])
    account_value.to_csv('./data/account_value.csv', index = False)
    
if __name__ == "__main__":
    main()