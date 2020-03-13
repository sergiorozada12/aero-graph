import numpy as np
import pandas as pd
import utils as u


N_SAMPLES = 3000

# DATA_PATH = "C:\\Users\\victor\\Documents\\Aero_TFG\\features\\"
# OUT_PATH = "C:\\Users\\victor\\Documents\\Aero_TFG\\modelIn\\"
DATA_PATH = '/home/server/Aero/features/'
OUT_PATH = '/home/server/Aero/modelIn/'

DATA_FILE_NODES = 'incoming_delays_nodes.csv'
DATA_FILE_ODS = 'incoming_delays_ods.csv'

######## READ THE DATA  ########
df_nodes = pd.read_csv(DATA_PATH + DATA_FILE_NODES, sep='|').fillna(0)
df_ods = pd.read_csv(DATA_PATH + DATA_FILE_ODS, sep='|').fillna(0)

avg_delays = pd.read_csv(DATA_PATH + 'avg_delays.csv', sep='|')

PERM_COLS = ['HOUR',
             'DAY',
             'DAY_OF_WEEK',
             'MONTH',
             'QUARTER',
             'SEASON',
             'MEAN_DELAY',
             'y_clas']

ALT_COLS = []
ALT_COLS += [col for col in df_nodes.columns if '_DELAY' in col] # Delays
ALT_COLS.remove('MEAN_DELAY')
ALT_COLS += list(avg_delays.columns)

nodes = np.array(sorted(df_nodes['NODE'].unique()))
od_pairs = np.array(sorted(df_ods['OD_PAIR'].unique()))


# OD PAIRS
u.feature_selection(df_ods, od_pairs, avg_delays,
                    PERM_COLS, ALT_COLS, N_SAMPLES,
                    'OD_PAIR', OUT_PATH)

# NODES
u.feature_selection(df_nodes, nodes, avg_delays,
                    PERM_COLS, ALT_COLS, N_SAMPLES,
                    'NODE', OUT_PATH)