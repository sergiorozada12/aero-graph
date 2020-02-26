import pandas as pd
import numpy as np

from utils import load_df_from_raw_files, cast_df_raw_columns, filter_airports, get_time_vars_od, get_time_vars_node, get_label

DATA_PATH = '/home/victor/Aero_TFG/Data'
OUT_PATH = '/home/victor/Aero_TFG/analysedData/'

COLUMNS = ['ARR_DELAY',
           'ARR_TIME',
           'CRS_ARR_TIME',
           'CRS_DEP_TIME',
           'DAY_OF_MONTH',
           'DAY_OF_WEEK',
           'DEP_DELAY',
           'DEP_TIME',
           'DEST',
           'FL_DATE',
           'MONTH',
           'ORIGIN',
           'QUARTER',
           'YEAR']

N_MOST_DELAYED = 100

TH = 60
H = 2

df_raw = load_df_from_raw_files(DATA_PATH)[COLUMNS].dropna()
df_casted = cast_df_raw_columns(df_raw)
df_filt, od_pairs, nodes = filter_airports(df_casted)

most_delayed_pairs = df_filt.groupby('OD_PAIR')\
                            .agg('mean')\
                            .sort_values(by='DEP_DELAY', ascending=False).index.values[:N_MOST_DELAYED]

df_filt['CRS_ARR_HOUR'] = df_filt['CRS_ARR_TIME'].apply(lambda x: int(x//100)).apply(lambda x: 0 if x == 24 else x)
df_filt['CRS_DEP_HOUR'] = df_filt['CRS_DEP_TIME'].apply(lambda x: int(x//100)).apply(lambda x: 0 if x == 24 else x)

df_filt['ARR_HOUR'] = df_filt['ARR_TIME'].apply(lambda x: int(x//100)).apply(lambda x: 0 if x == 24 else x)
df_filt['DEP_HOUR'] = df_filt['DEP_TIME'].apply(lambda x: int(x//100)).apply(lambda x: 0 if x == 24 else x)

######################################### OD-PAIRS ####################################################################

df_dep = pd.DataFrame(df_filt.groupby(['FL_DATE', 'CRS_DEP_HOUR', 'OD_PAIR'])['DEP_DELAY'].agg(list)).reset_index()
df_arr = pd.DataFrame(df_filt.groupby(['FL_DATE', 'CRS_ARR_HOUR', 'OD_PAIR'])['ARR_DELAY'].agg(list)).reset_index()

df_dep.columns = ['FL_DATE', 'HOUR', 'OD_PAIR', 'DEP_DELAY']
df_arr.columns = ['FL_DATE', 'HOUR', 'OD_PAIR', 'ARR_DELAY']

df_merged = df_dep.merge(df_arr, how='outer', on=['FL_DATE', 'HOUR', 'OD_PAIR'])

df_merged['ARR_DELAY'] = df_merged['ARR_DELAY'].apply(lambda x: x if type(x) == list else [])
df_merged['DEP_DELAY'] = df_merged['DEP_DELAY'].apply(lambda x: x if type(x) == list else [])

arr_delay_tm1 = df_merged['ARR_DELAY'].shift(od_pairs.shape[0]).apply(lambda x: x if type(x) == list else [])
dep_delay_tm1 = df_merged['DEP_DELAY'].shift(od_pairs.shape[0]).apply(lambda x: x if type(x) == list else [])

df_merged['ARR_DELAY'] = df_merged['ARR_DELAY'] + arr_delay_tm1
df_merged['DEP_DELAY'] = df_merged['DEP_DELAY'] + dep_delay_tm1

df_merged['MEAN_ARR_DELAY'] = df_merged['ARR_DELAY'].apply(np.mean)
df_merged['MEAN_DEP_DELAY'] = df_merged["DEP_DELAY"].apply(np.mean)

df_merged['FL_DATE'] = pd.to_datetime(df_merged['FL_DATE'])

dates = pd.to_datetime(df_filt['FL_DATE'].unique()).sort_values()
hours = np.sort(df_filt['CRS_ARR_HOUR'].unique())

df_complete = get_time_vars_od(df_merged, dates, hours, od_pairs).drop(columns=['MONTH'], inplace=True)
df_final = get_label(df_complete, TH, H, od_pairs)
df_final.to_csv(OUT_PATH + 'signal.csv', sep='|', index=False, index_label=False)

######################################### AIRPORTS ####################################################################

df_origin = pd.DataFrame(df_filt.groupby(['FL_DATE', 'CRS_DEP_HOUR', 'ORIGIN'])['DEP_DELAY'].agg(list)).reset_index()
df_arr = pd.DataFrame(df_filt.groupby(['FL_DATE', 'CRS_ARR_HOUR', 'DEST'])['ARR_DELAY'].agg(list)).reset_index()

df_origin.columns = ['FL_DATE', 'HOUR', 'NODE', 'DEP_DELAY']
df_arr.columns = ['FL_DATE', 'HOUR', 'NODE', 'ARR_DELAY']

df_merged_node = df_origin.merge(df_arr, how='outer', on=['FL_DATE', 'HOUR', 'NODE'])
df_merged_node['ARR_DELAY'] = df_merged['ARR_DELAY'].apply(lambda x: x if type(x) == list else [])
df_merged_node['DEP_DELAY'] = df_merged['DEP_DELAY'].apply(lambda x: x if type(x) == list else [])

arr_delay_tm1 = df_merged_node['ARR_DELAY'].shift(nodes.shape[0]).apply(lambda x: x if type(x) == list else [])
dep_delay_tm1 = df_merged_node['DEP_DELAY'].shift(nodes.shape[0]).apply(lambda x: x if type(x) == list else [])

df_merged_node['ARR_DELAY'] = df_merged_node['ARR_DELAY'] + arr_delay_tm1
df_merged_node['DEP_DELAY'] = df_merged_node['DEP_DELAY'] + dep_delay_tm1

df_merged_node['MEDIAN_NODE_DELAY'] = (df_merged_node["DEP_DELAY"] + df_merged_node["ARR_DELAY"]).apply(np.median)
df_merged_node['MEDIAN_NODE_DEP_DELAY'] = df_merged_node['DEP_DELAY'].apply(np.median)
df_merged_node['MEDIAN_NODE_ARR_DELAY'] = df_merged_node['ARR_DELAY'].apply(np.median)
df_merged_node['FL_DATE'] = pd.to_datetime(df_merged_node['FL_DATE'])

df_complete = get_time_vars_node(df_merged_node, dates, hours, nodes)

df_com_nodes = df.merge(df_merged_node, how='left', on=['FL_DATE', 'HOUR', 'NODE'])
df_com_nodes.drop(columns=['DEP_DELAY', 'ARR_DELAY'])\
    .to_csv(OUT_PATH + 'airport_delays.csv', sep='|', index=False, index_label=False)
