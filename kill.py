#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 23:30:05 2022

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
    api, account_value = login()
    print('Account value at kill start ${}'.format(account_value))
    close_trades(api)
    api, account_value = login()
    print('\nAll trades and orders killed, account value after kill procedure %a' %account_value)
    

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
    
    account_value = account.cash 
    
    return api, account_value

def close_trades(api):
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

if __name__ == "__main__":
    main()