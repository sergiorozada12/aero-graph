import pandas as pd
import numpy as np

import utils as u

DATA_PATH = '/home/server/Aero/aero-graph/raw_data/'
OUT_PATH = '/home/server/Aero/data/'
# DATA_PATH = "C:\\Users\\victor\\Documents\\Aero_TFG\\raw_data\\"
# OUT_PATH = "C:\\Users\\victor\\Documents\\Aero_TFG\\data\\"

N_MOST_DELAYED = 100

TH = 60
H = 2

TIME_COLS = ['FL_DATE',
             'HOUR',
             'DAY',
             'DAY_OF_WEEK',
             'MONTH',
             'QUARTER',
             'SEASON',
             'YEAR']

DELAY_COLS = ['MEDIAN_DELAY',
              'MEAN_DELAY',
              'MEAN_DEP_DELAY',
              'MEAN_ARR_DELAY',
              'MEDIAN_CARRIER_DELAY',
              'MEAN_CARRIER_DELAY',
              'MEDIAN_LATE_AIRCRAFT_DELAY',
              'MEAN_LATE_AIRCRAFT_DELAY',
              'MEDIAN_NAS_DELAY',
              'MEAN_NAS_DELAY',
              'MEDIAN_SECURITY_DELAY',
              'MEAN_SECURITY_DELAY',
              'MEDIAN_WEATHER_DELAY',
              'MEAN_WEATHER_DELAY']


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

print("OD-PAIRS pipeline")
print("----------------------------------------")
df_merged_od_pairs = u.merge_and_group(df_filt_od_pairs, problem_type=u.OD_PAIR)
df_od_pairs = u.obtain_avg_delay(df_merged_od_pairs, shift=od_pairs.shape[0])
df_od_pairs = u.get_time_vars(df_od_pairs, dates, hours, od_pairs, 'OD_PAIR')
df_final_od_pairs = u.get_label(df_od_pairs, TH, H, od_pairs.shape[0])
df_final_od_pairs[TIME_COLS + DELAY_COLS + ['OD_PAIR', 'y_clas']].to_csv(OUT_PATH + 'dataset_od_pairs.csv', sep='|', index=False, index_label=False)
u.create_od_pair_graph(od_pairs, OUT_PATH)
print("DONE")
print("----------------------------------------")

print("AIRPORTS pipeline")
print("----------------------------------------")
df_merged_node = u.merge_and_group(df_filt_airports, problem_type=u.NODE)
df_nodes = u.obtain_avg_delay(df_merged_node, shift=nodes_airports.shape[0])
df_nodes = u.get_time_vars(df_nodes, dates, hours, nodes_airports, 'NODE')
df_final_nodes = u.get_label(df_nodes, TH, H, nodes_airports.shape[0])

df_final_nodes[TIME_COLS + DELAY_COLS + ['NODE', 'y_clas']].to_csv(OUT_PATH + 'dataset_airports.csv', sep='|', index=False, index_label=False)
u.create_airport_graph(df_filt_airports, nodes_airports, OUT_PATH)

print("DONE")
print("----------------------------------------")