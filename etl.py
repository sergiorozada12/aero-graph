import pandas as pd
import numpy as np

import utils as u

DATA_PATH = '/home/victor/Aero_TFG/Data'
OUT_PATH = '/home/victor/Aero_TFG/analysedData/'

N_MOST_DELAYED = 100

TH = 60
H = 2

df_raw = u.load_df_from_raw_files(DATA_PATH).dropna()
df_casted = u.cast_df_raw_columns(df_raw)
df_filt_od_pairs, od_pairs, nodes_od_pairs = u.filter_od_pairs(df_casted)
df_filt_airports, nodes_airports = u.filter_airports(df_casted)

most_delayed_pairs = df_filt_od_pairs.groupby('OD_PAIR')\
                                     .agg('mean')\
                                     .sort_values(by='DEP_DELAY', ascending=False).index.values[:N_MOST_DELAYED]

df_filt_od_pairs = u.get_hour(df_filt_od_pairs)
df_filt_airports = u.get_hour(df_filt_airports)

dates = pd.to_datetime(df_filt_od_pairs['FL_DATE'].unique()).sort_values()
hours = np.sort(df_filt_od_pairs['CRS_ARR_HOUR'].unique())

# OD-PAIRS
df_merged_od_pairs = u.merge_and_group(df_filt_od_pairs, shift=od_pairs.shape[0], problem_type=0)
df_od_pairs = u.get_time_vars(df_merged_od_pairs, dates, hours, od_pairs.shape[0]).drop(columns=['MONTH'], inplace=True)
df_final_od_pairs = u.get_label(df_od_pairs, TH, H, od_pairs.shape[0])
df_final_od_pairs.to_csv(OUT_PATH + 'signal.csv', sep='|', index=False, index_label=False)

# AIRPORTS
df_merged_node = u.merge_and_group(df_filt_airports, shift=nodes_airports.shape[0], problem_type=1)
df_nodes = u.get_time_vars(df_merged_node, dates, hours, nodes_airports.shape[0])
df_final_nodes = u.get_label(df_nodes, TH, H, nodes_airports.shape[0])
df_final_nodes.drop(columns=['DEP_DELAY', 'ARR_DELAY'])\
    .to_csv(OUT_PATH + 'airport_delays.csv', sep='|', index=False, index_label=False)
