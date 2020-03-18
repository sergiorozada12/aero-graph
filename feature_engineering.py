import numpy as np
import pandas as pd
import utils as u


DATA_PATH = '/home/server/Aero/data/'
OUT_PATH = '/home/server/Aero/features/'
# DATA_PATH = "C:\\Users\\victor\\Documents\\Aero_TFG\\data\\"
# OUT_PATH = "C:\\Users\\victor\\Documents\\Aero_TFG\\features\\"

DELAY_TYPES = ['DELAY', 'CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']

df = pd.read_csv(DATA_PATH + 'dataset_od_pairs.csv', sep='|')
df_casted = u.cast_df_raw_columns(df)

print("Read OD Pairs data with shape " + str(df.shape))

df_nodes = pd.read_csv(DATA_PATH + 'dataset_airports.csv', sep='|')

print("Read Airports data with shape " + str(df_nodes.shape))

nodes = np.array(sorted(df_nodes['NODE'].unique()))
od_pairs = np.array(sorted(df_casted['OD_PAIR'].unique()))
dates = np.array(sorted(df_casted['FL_DATE'].unique()))
hours = np.array(sorted(df_casted['HOUR'].unique()))

df_casted.drop(columns=['FL_DATE'], inplace=True)

df_avg_delays = u.get_features_df(df_casted, df_nodes, od_pairs, nodes)

df_d_types = u.get_delay_types_df(df_casted, df_nodes, DELAY_TYPES, od_pairs, nodes)
print(df_d_types.shape)

assert len(df_avg_delays) ==\
       dates.shape[0]*hours.shape[0],\
       "Dataframes doesn't have the correct length"


######  INTERPOLATION   ######
LIMIT_INT = 6
NO_INT_HOUR = 4

print("----------------------------------------")
print("Interpolation - ", end="")

df_avg_delays = df_avg_delays.interpolate(limit=LIMIT_INT)
df_avg_delays.fillna(0, inplace=True)

df_d_types = df_d_types.interpolate(limit=LIMIT_INT)
df_d_types.fillna(0, inplace=True)

print("DONE")

######  CLUSTERING   ######
n_clust = 6
random_state = 42

print("----------------------------------------")
print("Clustering - ", end="")

df_avg_delays = u.apply_clustering(df_avg_delays, n_clust, dates.shape[0], hours.shape[0], random_state)

print("DONE")

#########     ADDING NEIGHBOUR DELAY INFORMATION        ################

print("----------------------------------------")
print("Obtaining neighbour information")

# Load edges
edges_nodes = np.load(DATA_PATH + "graph/edges_nodes.npy", allow_pickle=True)
edges_od_pairs = np.load(DATA_PATH + "graph/edges_od_pairs.npy", allow_pickle=True)

colums_incoming_delay = []

for d in DELAY_TYPES:
    print("Working on " + d)
    col = 'INCOMING_MEAN_' + d
    # OD Pairs
    print("OD Pairs")
    df_casted[col] = u.treat_delay_type(d, df_casted[['OD_PAIR', 'MEAN_' + d]], od_pairs, 'OD_PAIR', edges_od_pairs)

    # Nodes
    print("Nodes")
    df_nodes[col] = u.treat_delay_type(d, df_nodes[['NODE', 'MEAN_' + d]], nodes, 'NODE', edges_nodes)

    colums_incoming_delay.append(col)

    print("DONE")

######  SAVING THE DATA   ######

print("----------------------------------------")
print("Saving the data")

df_avg_delays.to_csv(OUT_PATH + 'avg_delays.csv', sep='|', index=False, index_label=False)
df_d_types.to_csv(OUT_PATH + 'df_d_types.csv', sep='|', index=False, index_label=False)
df_casted.to_csv(OUT_PATH + 'incoming_delays_ods.csv', sep='|', index=False, index_label=False)
df_nodes.to_csv(OUT_PATH + 'incoming_delays_nodes.csv', sep='|', index=False, index_label=False)