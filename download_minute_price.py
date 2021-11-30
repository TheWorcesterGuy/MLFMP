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

warnings.simplefilter(action='ignore')

import yaml

def main():

	stocks = ['INTC', 'TSLA',  'AMZN', 'FB', 'AAPL', 'DIS', 'SPY', 'QQQ', 'GOOG', 'GOOGL', 'MSFT', 'NFLX', 'NVDA', 'TWTR', 'AMD', 'WMT', 'JPM', 'BAC', 'PG']
              

	with open('credentials.yaml', 'r') as stream:
		creds = yaml.safe_load(stream)

		alpaca_key_id = creds['alpaca']['key_id']
		alpaca_secret_key = creds['alpaca']['secret_key']


	for stock in stocks:
		print('\nDownloading minute market price data for %s...\n' % stock)

		nb_file = len(glob.glob('./data/TRADE_DATA/minute_data/%s.csv' % stock))
		if nb_file > 0:
			df = pd.read_csv('./data/TRADE_DATA/minute_data/%s.csv' % stock)
			df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
			df = df.sort_values('Datetime')
			start_date = df['Datetime'].iloc[-400]  # to make sure we don't have any hole
		else:
			start_date = datetime(2014, 1, 1)

		price_data = get_price(stock, alpaca_key_id, alpaca_secret_key, start_date)
		price_data = price_data.reset_index(drop=False)
		price_data = price_data.rename(columns={'timestamp': 'Datetime'})

		if nb_file > 0:
			historical_data = pd.read_csv('./data/TRADE_DATA/minute_data/%s.csv' % stock)
			price_data = historical_data.append([price_data])

		price_data = price_data.drop_duplicates(subset=['Datetime'], keep='first')
		price_data.to_csv('./data/TRADE_DATA/minute_data/%s.csv' % stock, index=False)
		print('\nDone downloading market minute price data...\n')


def get_price(ticker, alpaca_key_id, alpaca_secret_key, start_date):
	APCA_API_KEY_ID = alpaca_key_id
	APCA_API_SECRET_KEY = alpaca_secret_key
	APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
	APCA_API_DATA_URL = 'https://data.alpaca.markets'
	APCA_RETRY_MAX = 3
	APCA_RETRY_WAIT = 3
	APCA_RETRY_CODES = 429.504
	api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL)

	temp = api.get_bars(ticker, TimeFrame.Minute, start_date.strftime('%Y-%m-%d'),
                        (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), limit=10000000000,
                        adjustment='raw').df

	time.sleep(1)

	return temp


if __name__ == "__main__":
	main()
