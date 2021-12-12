#!/usr/bin/env python3


import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timezone
import os.path
import glob
import sys
import os
import numpy as np
import time


keyword = sys.argv[1]


def main():

    files = glob.glob('data/TWITTER_DATA/%s/*.csv' % keyword)
    files = [file for file in files if 'encoded' not in file]

    if len(files) > 0:
        list_tw = [pd.read_csv(file, lineterminator='\n')[['Datetime', 'Text', 'retweetCount', 'likeCount', 'nbFollowers']]
                   for file in
                   glob.glob('data/TWITTER_DATA/%s/*.csv' % keyword) if 'encoded' not in file]
        df = pd.concat(list_tw, axis=0)
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df['date'] = df['Datetime'].dt.date
        df = df.drop_duplicates(['date', 'Text']).sort_values('Datetime')
        del df['date']
        df = df.sort_values('Datetime')
        start_date = df['Datetime'].max() - pd.DateOffset(hours=2)  # to make sure we don't have any holes
    else:
        start_date = pd.to_datetime('2016-01-01', format='%Y-%m-%d')
        df = pd.DataFrame()

    final_date = datetime.utcnow().replace(tzinfo=timezone.utc)
    start_date = start_date.replace(tzinfo=timezone.utc)
    end_date = start_date

    print("\nDownloading Tweets for %s..." % (keyword.upper()))

    while final_date > end_date:

        delta_days = (final_date - start_date).days

        if delta_days > 30:
            delta_days = 30

        end_date = start_date + pd.DateOffset(days=delta_days + 2)

        # download tweets with API query
        tweets_df = download(keyword, start_date, end_date, 10)

        df = pd.concat([df, tweets_df], axis=0)
        df['date'] = df['Datetime'].dt.date
        df = df.drop_duplicates(['date', 'Text']).sort_values('Datetime')
        del df['date']

        start_date = pd.Timestamp(df['Datetime'].max() - pd.DateOffset(days=1)).replace(tzinfo=timezone.utc)
        final_date = datetime.utcnow().replace(tzinfo=timezone.utc)

    df['year'] = df['Datetime'].apply(lambda x: x.isocalendar()[0])
    df['week'] = df['Datetime'].apply(lambda x: x.isocalendar()[1])

    years = df['year'].unique()
    weeks = df['week'].unique()

    for year in years:
        df_year = df[df['year'] == year]
        for week in weeks:
            df_year_week = df_year[df_year['week'] == week][['Datetime', 'Text', 'retweetCount', 'likeCount', 'nbFollowers']]
            if df_year_week.empty is False:
                df_year_week.to_csv('data/TWITTER_DATA/%s/tweets_%s_%s_%s.csv' % (keyword, keyword, year, week), index=False)
    print('Done downloading %s ' % keyword)


def download(stock, start_date, end_date, tries):
    for i in range(tries):
        try:
            tweets_list = []

            # Using TwitterSearchScraper to scrape data and append tweets to list
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper('%s since:%s-%s-%s until:%s-%s-%s' %
                                                                     (
                                                                     '$%s' % keyword, start_date.year, start_date.month,
                                                                     start_date.day, end_date.year, end_date.month,
                                                                     end_date.day)).get_items()):

                if tweet.lang == 'en' and tweet.content.count(
                        '$') < 3:  # and '#' not in tweet.content and 'http' not in tweet.content and '@' not in tweet.content:
                    tweets_list.append(
                        [tweet.date, tweet.content, tweet.retweetCount, tweet.likeCount, tweet.user.followersCount])

            tweets_df = pd.DataFrame(tweets_list,
                                     columns=['Datetime', 'Text', 'retweetCount', 'likeCount', 'nbFollowers'])

            return tweets_df

        except:
            if i == tries - 1:
                raise ValueError('Failed to download tweets for stock %s after %s retries' % (stock, tries))
            time.sleep(2)
            continue


if __name__ == "__main__":
    main()
