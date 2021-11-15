import pandas as pd
import numpy as np
from datetime import datetime
pd.options.mode.chained_assignment = None

current_date = datetime.today().strftime('%Y-%m-%d')

df = pd.read_csv('./data/features_store.csv')

# take latest rows
df_last_day = df[df['Date'] == df['Date'].max()]

# compute metrics for latest rows
df_metric = df_last_day[['Date', 'stock']]
df_metric['number_missing_values'] = df_last_day.isnull().sum(axis=1)

df_metric['percentage_missing_values'] = np.round(df_metric['number_missing_values'] / df_last_day.shape[1] * 100, 2)

# compute global metric to compare against
df_sample = df.iloc[-500:]
df_metric_sample = df_sample[['Date', 'stock']]
df_metric_sample['mean_number_missing_values'] = df_sample.isnull().sum(axis=1)
df_metric_sample['mean_percentage_missing_values'] = np.round(df_metric_sample['mean_number_missing_values'] / df_sample.shape[1] * 100, 2)

df_metric_sample = df_metric_sample.groupby('stock').mean()
df_metric_sample['mean_number_missing_values'] = np.round(df_metric_sample['mean_number_missing_values'], 2)
df_metric_sample['mean_percentage_missing_values'] = np.round(df_metric_sample['mean_percentage_missing_values'], 2)

# join and save
df = df_metric.merge(df_metric_sample, on='stock', how='outer')
df.to_csv('./log/%s_features_store_log.csv' % current_date, index=False)