import numpy as np
import pandas as pd
from utils import cast_df_raw_columns

from sklearn.cluster import KMeans

DATA_PATH = 'C:/Users/victor/Aero_TFG/data/'
OUT_PATH = 'C:/Users/victo/Aero_TFG/modelIn/'

df = pd.read_csv(DATA_PATH + 'dataset_od_pairs.csv', sep='|')
df_casted = cast_df_raw_columns(df).drop(columns=['DEP_DELAY', 'ARR_DELAY'])

df_nodes = pd.read_csv(DATA_PATH + 'dataset_airports.csv', sep='|')

nodes = np.array(sorted(df_nodes['NODE'].unique()))
od_pairs = np.array(sorted(df_casted['OD_PAIR'].unique()))
dates = np.array(sorted(df_casted['FL_DATE'].unique()))
hours = np.array(sorted(df_casted['HOUR'].unique()))

df_casted.drop(columns=['FL_DATE'], inplace=True)

avg_delays_od_pairs = df['MEDIAN_DEP_DELAY'].values.reshape(-1, od_pairs.shape[0]).astype(np.float32)
df_od_med_delays = pd.DataFrame(avg_delays_od_pairs, columns=od_pairs)

avg_delays_nodes_dep = df_nodes['MEDIAN_NODE_DEP_DELAY'].values.reshape(-1, nodes.shape[0]).astype(np.float32)
avg_delays_nodes_arr = df_nodes['MEDIAN_NODE_ARR_DELAY'].values.reshape(-1, nodes.shape[0]).astype(np.float32)

cols_dep = [n + '_DEPAR' for n in nodes]
cols_arr = [n + '_ARRIV' for n in nodes]

df_node_med_delays = pd.concat([pd.DataFrame(avg_delays_nodes_dep, columns=cols_dep),
                                pd.DataFrame(avg_delays_nodes_arr, columns=cols_arr)], axis=1)

df_med_delays = pd.concat([df_od_med_delays, df_node_med_delays], axis=1)

assert len(df_od_med_delays) ==\
       len(df_node_med_delays) ==\
       len(df_med_delays) ==\
       dates.shape[0]*hours.shape[0],\
       "Dataframes doesn't have the correct length"


######  INTERPOLATION   ######
LIMIT_INT = 6
NO_INT_HOUR = 4

df_med_delays_int = df_med_delays.interpolate(limit=LIMIT_INT)
df_med_delays_int.fillna(0, inplace=True)


######  CLUSTERING   ######
n_clust = 6
random_state = 42

kmeans_h = KMeans(n_clusters=n_clust, random_state=random_state)
kmeans_h.fit(df_med_delays_int.values)
df_med_delays_int['HOUR_CLUSTER'] = kmeans_h.labels_

kmeans_d = KMeans(n_clusters=n_clust, random_state=random_state)
kmeans_d.fit(df_med_delays_int.values.reshape(dates.shape[0], -1))
df_med_delays_int['DAY_CLUSTER'] = np.repeat(kmeans_d.labels_, hours.shape[0])

df_med_delays_int['DAY_CLUSTER-1'] = df_med_delays_int['DAY_CLUSTER'].shift(-hours.shape[0], fill_value=1)


######  SAVING THE DATA   ######

df_med_delays_int.to_csv(OUT_PATH + 'df_med_delays.csv', sep='|')