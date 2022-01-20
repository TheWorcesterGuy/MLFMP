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
import yaml
warnings.simplefilter(action = 'ignore')

def main():
    print('\nIn Alpaca Trading\n')
    nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
    api, buying_power, account_value = login()
    print('\nLogin test --- Completed\n')
    trade = pd.read_csv('./data/to_trade.csv')
    trade_data, trade_value = distribution(api, buying_power)
    print('\nMoney allocation --- Completed\n')
    open_trade(api, trade_data)
    os.system('python email_updates_morning.py')
    print('\nTrading--- Completed\n')
    nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
    close = nyc_datetime.replace(hour=15, minute=59, second=0,microsecond=0)
    difference = close - nyc_datetime
    print('\nSleeping before market close : %a hours\n'%round((difference.seconds/60)/60,2))
    time.sleep(difference.seconds)
    print('\nClosing trades\n')
    close_trades(api, account_value, trade_value)
    print('\nTrades closed\n')
            

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
    
    with open('credentials.yaml', 'r') as stream:
        creds = yaml.safe_load(stream)

        alpaca_key_id = creds['alpaca']['key_id']
        alpaca_secret_key = creds['alpaca']['secret_key']
    
    APCA_API_KEY_ID = alpaca_key_id
    APCA_API_SECRET_KEY = alpaca_secret_key
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

def distribution(api, buying_power) :
    """Class function used to distribute a reduced buying power between stocks based on probabilities and number of predictions'

    Returned
    --------
    `df` : Pandas DataFrame
        Contains the final trading data, note this data is also exported to '.csv' for email updates
    """
    account = api.get_account()
    reduced_power = (float(account.buying_power)/4)
    reduced_power = reduced_power - reduced_power*0.01
    print('The trading power today is: %a $' %reduced_power)
    df = pd.read_csv('./data/to_trade.csv')
    
    Last_Stock_Value = []
    Fractionable = []
    Variability = []
    
    for stock in df['Products'] :
        barset = api.get_barset(stock, 'day', limit=5)
        bars = barset[stock]
        Last_Stock_Value.append(bars[-1].c + bars[-1].c * 0.02)
        side = df['Side'] [df['Products'] == stock].iloc[0]
        
        if side < 0 :
            frac = False
        else :
            frac = api.get_asset(stock).fractionable
        
        Fractionable.append(frac)
        Variability.append(stock_variability(stock))
        
         
    df['Fractionable'] = Fractionable
    df['Variability'] = Variability
    df['Last_Stock_Value_$'] = Last_Stock_Value
    df['K%'] = df['K%']/df['Variability']
    df['Value_to_trade_$'] = (df['K%']*reduced_power)
    df = df.sort_values(by=['Value_to_trade_$'], ascending=False)
    df['Quantity'] = df['Value_to_trade_$']/df['Last_Stock_Value_$']
    
    df['Quantity'][df['Fractionable'] == False] = np.floor(df['Quantity'][df['Fractionable'] == False].tolist())
    df['Quantity'][df['Fractionable'] == True] = np.around(df['Quantity'][df['Fractionable'] == True].tolist(),2)
    df = df[df['Quantity'] > 0]
    trade_value = df['Value_to_trade_$'].sum()
    
    print(df)
    df.to_csv('./data/days_trades.csv', index = False)
    
    return df, trade_value    

def stock_variability(stock):
    SPY = pd.read_csv('./data/TRADE_DATA/price_data/SPY.csv')
    SPY = SPY.iloc[-100:]
    SPY = ((SPY['Close']-SPY['Open'])/SPY['Open']).abs().mean()
    
    df = pd.DataFrame()
    temp = pd.read_csv('./data/TRADE_DATA/price_data/' + stock + '.csv')
    temp = temp.iloc[-100:]
    temp = ((temp['Close']-temp['Open'])/temp['Open']).abs().mean()/SPY
    return np.round(temp,2)

def open_trade(api, trade_data) :
    """Function used to open trades based on the stock, trade type and quantity'

    """
    
    positions = api.list_positions()
    orders = api.list_orders(status='open')
    
    print('Postions : If the following detail is not empty then, trade warning')
    print(positions)
    
    print('Orders : If the following detail does not change then, trade warning')
    print(orders)
    
    stocks = trade_data['Products'].tolist()
    sides = trade_data['Side'].tolist()
    quantities = trade_data['Quantity'].tolist()
    
    for stock,side,quantity in zip(stocks,sides,quantities) :
    
        if side >= 1 :
            api.submit_order(
                symbol = stock,
                qty = quantity,  
                side = 'buy',
                type = 'market',
                time_in_force = 'day',
            )
            
        if side <= -1 :
            api.submit_order(
                symbol = stock,
                qty = quantity,  
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
                
    time.sleep(900)    
    account = api.get_account()
    cash = account.cash
    account_value = pd.read_csv('./data/account_value.csv')
    df = pd.DataFrame({'Date' : datetime.now().strftime('%Y - %m - %d'), 'PM' : [cash], 'Trade_value' : [trade_value]})
    account_value = account_value.merge(df, on='Date')
    account_value['Change_account_%'] = 100*(float(account_value ['PM'].iloc[0]) - float(account_value ['AM'].iloc[0]))/float(account_value ['AM'].iloc[0])
    account_value.to_csv('./data/account_value.csv', index = False)
    
if __name__ == "__main__":
    main()