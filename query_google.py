import pandas as pd
import numpy as np
from pytrends import dailydata
from datetime import datetime
from pytrends.request import TrendReq
import pytrends
from pathlib import Path
import time
import random
import urllib.request
import matplotlib.pyplot as plt
import sys
import pytz


args = sys.argv[1:]

year_start = int(args[0])
month_start = int(args[1])
day_start = int(args[2])
year_end = int(args[3])
month_end = int(args[4])
day_end = int(args[5])


keywords = args[6:]
keyword = ' '.join(keywords)

time_ref = pd.date_range("%s-%s-%s 00:00:00" % (year_start, month_start, day_start),
                         "%s-%s-%s 00:00:00" % (year_end, month_end, day_end), freq="1H") \
                                .to_frame(name='date').reset_index(drop=True)

pytrends = TrendReq(hl='en-US', geo='', tz=360, retries=0, timeout=(30, 45))

df = pytrends.get_historical_interest([keyword], year_start=year_start, month_start=month_start,
                                      day_start=day_start,
                                      hour_start=0, year_end=year_end, month_end=month_end,
                                      day_end=day_end,
                                      hour_end=0, cat=7, geo='', gprop='')

if df.empty:
    df = pd.DataFrame(columns=['date', keyword, 'isPartial'])

df = df.merge(time_ref, on='date', how='right')[['date', keyword, 'isPartial']]

df[keyword] = df[keyword].fillna(0)
df['isPartial'] = df['isPartial'].fillna('False')

df['isPartial'] = df['isPartial'].astype(str)
df.loc[((df['isPartial'] == 'True') & (df[keyword] == 0)) | ((df['isPartial'] == 'True') & (df.index != df.index.max())), 'isPartial'] = 'Suspect'
end_date = df.index[(df['isPartial'] == 'True')].tolist()

if len(end_date) > 0:
    df = df.iloc[:end_date[0] + 1]

# convert to NY time (date as index already)
eastern = pytz.timezone('US/Eastern')
df.set_index('date', inplace=True)
df.index = df.index.tz_localize(pytz.utc).tz_convert(eastern)

df = df.sort_index()
df = df.reset_index()

nb_missing_values = df[df[keyword] == 0].shape[0]
percent_missing_value = np.round(nb_missing_values / df.shape[0] * 100, 2)

print('Number of missing values : %s (%s%%)' % (nb_missing_values, percent_missing_value))

df.to_csv('data/temp_data.csv', index=False)
