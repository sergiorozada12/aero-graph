import pandas as pd
import numpy as np

import utils as u

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

df_raw = u.load_df_from_raw_files(DATA_PATH)[COLUMNS].dropna()
df_casted = u.cast_df_raw_columns(df_raw)
df_filt, od_pairs, nodes = u.filter_airports(df_casted)

most_delayed_pairs = df_filt.groupby('OD_PAIR')\
                            .agg('mean')\
                            .sort_values(by='DEP_DELAY', ascending=False).index.values[:N_MOST_DELAYED]

df_filt = u.get_hour(df_filt)

dates = pd.to_datetime(df_filt['FL_DATE'].unique()).sort_values()
hours = np.sort(df_filt['CRS_ARR_HOUR'].unique())

# OD-PAIRS
df_merged_od_pairs = u.merge_and_group(df_filt, shift=od_pairs.shape[0], problem_type=0)
df_od_pairs = u.get_time_vars_od(df_merged_od_pairs, dates, hours, od_pairs).drop(columns=['MONTH'], inplace=True)
df_final_od_pairs = u.get_label(df_od_pairs, TH, H, od_pairs)
df_final_od_pairs.to_csv(OUT_PATH + 'signal.csv', sep='|', index=False, index_label=False)

# AIRPORTS
df_merged_node = u.merge_and_group(df_filt, shift=nodes.shape[0], problem_type=1)
df_nodes = u.get_time_vars_node(df_merged_node, dates, hours, nodes)
df_com_nodes.drop(columns=['DEP_DELAY', 'ARR_DELAY'])\
    .to_csv(OUT_PATH + 'airport_delays.csv', sep='|', index=False, index_label=False)
