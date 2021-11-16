import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import os
pd.options.mode.chained_assignment = None

current_date = datetime.today().strftime('%Y-%m-%d')

df = pd.read_csv('./data/features_store.csv')
healthy_feature_store = True

# take latest rows
df_last_day = df[df['Date'] == df['Date'].max()]

# compute metrics for latest rows
df_metric = df_last_day[['Date', 'stock']]
df_metric['nb_nans'] = df_last_day.isnull().sum(axis=1)

df_metric['percentage_nans'] = np.round(df_metric['nb_nans'] / df_last_day.shape[1] * 100, 2)

# compute global metric to compare against
df_sample = df.iloc[-500:]
df_metric_sample = df_sample[['Date', 'stock']]
df_metric_sample['mean_nb_nans'] = df_sample.isnull().sum(axis=1)
df_metric_sample['mean_percentage_nans'] = np.round(df_metric_sample['mean_nb_nans'] / df_sample.shape[1] * 100, 2)

df_metric_sample = df_metric_sample.groupby('stock').mean()
df_metric_sample['mean_nb_nans'] = np.round(df_metric_sample['mean_nb_nans'], 2)
df_metric_sample['mean_percentage_nans'] = np.round(df_metric_sample['mean_percentage_nans'], 2)

# join and save
df = df_metric.merge(df_metric_sample, on='stock', how='outer')

my_file = Path("./data/top_50.csv")
if my_file.is_file():
    df_features = pd.read_csv('./data/top_50.csv')
    top_features = ['stock'] + df_features['Features'].values.tolist()

    df_top_features = df_last_day[top_features]
    df_sample_top_features = df_sample[top_features]

    df_top_features['nb_nans_top_50'] = df_top_features.isnull().sum(axis=1)
    df_sample_top_features['mean_nb_nans_top_50'] = df_sample_top_features.isnull().sum(axis=1)
    df_sample_top_features = df_sample_top_features[['stock', 'mean_nb_nans_top_50']].groupby('stock').mean()
    df_sample_top_features['mean_nb_missing_values_top_50'] = np.round(df_sample_top_features['mean_nb_nans_top_50'], 2)

    df_top_50 = df_top_features.merge(df_sample_top_features, on='stock', how='outer')
    df_top_50 = df_top_50[['stock', 'nb_nans_top_50', 'mean_nb_nans_top_50']]

    df = df_top_50.merge(df, on='stock', how='outer')[['Date', 'nb_nans_top_50', 'mean_nb_nans_top_50',
                                                       'nb_nans', 'mean_nb_nans', 'percentage_nans', 'mean_percentage_nans']]

    if df['nb_nans_top_50'].sum() > 0:
        healthy_feature_store = False

# write report in log
df.to_csv('./log/%s_features_store_log.csv' % current_date, index=False)

# if last date in feature store is not today's
if df['Date'].max() != current_date:
    healthy_feature_store = False

# delete feature store if healthy check failed
if not healthy_feature_store :
    os.system('rm ./data/features_store.csv')

