import numpy as np
import pandas as pd
from utils import cast_df_raw_columns

from sklearn.cluster import KMeans

DATA_PATH = '/home/victor/Aero_TFG/analysedData/'
OUT_PATH = '/home/victor/Aero_TFG/modelIn/'

df = pd.read_csv(DATA_PATH + 'signal.csv', sep='|')
df_casted = cast_df_raw_columns(df).drop(columns=['DEP_DELAY', 'ARR_DELAY'])

od_pairs = np.array(sorted(df_casted['OD_PAIR'].unique()))
dates = np.array(sorted(df_casted['FL_DATE'].unique()))
hours = np.array(sorted(df_casted['HOUR'].unique()))

df_casted.drop(columns=['FL_DATE'], inplace=True)

avg_delays_od_pairs = df['MEDIAN_DEP_DELAY'].values.reshape(-1, od_pairs.shape[0]).astype(np.float32)
df_od_med_delays = pd.DataFrame(avg_delays_od_pairs, columns=od_pairs)

avg_delays_nodes_dep = df_nodes['MEDIAN_NODE_DEP_DELAY'].values.reshape(-1, nodes.shape[0]).astype(np.float32)
avg_delays_nodes_arr = df_nodes['MEDIAN_NODE_ARR_DELAY'].values.reshape(-1, nodes.shape[0]).astype(np.float32)
