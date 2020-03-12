import numpy as np
import pandas as pd
import json
import utils as u


N_SAMPLES = 3000
COL = 'NODE'

SELECTOR_RF = 1

OUT_PATH = 'C:/Users/E054031/Desktop/phd/3 - research/0 - aero/paper_victor/output_data/'
DATA_PATH = "E:/TFG_VictorTenorio/Aero_TFG/features/"
OUT_PATH = "E:/TFG_VictorTenorio/Aero_TFG/modelIn/"

DATA_FILE = 'incoming_delays_nodes.csv'
OUT_FILE = 'features_lr.json'


######## READ THE DATA  ########
df = pd.read_csv(DATA_PATH + DATA_FILE, sep='|').fillna(0)

avg_delays = pd.read_csv(DATA_PATH + 'avg_delays.csv', sep='|')

cols_data = list(df.columns) + list(avg_delays.columns)
cols_data.remove('y_clas')
cols_data.remove('NODE')
cols_data.remove('FL_DATE')

PERM_COLS = ['HOUR',
             'DAY',
             'DAY_OF_WEEK',
             'MONTH',
             'QUARTER',
             'SEASON',
             'MEAN_DELAY',
             'y_clas']

ALT_COLS = []
ALT_COLS += [col for col in df.columns if '_DELAY' in col] # Delays
ALT_COLS.remove('MEAN_DELAY')
ALT_COLS += list(avg_delays.columns)

entities = np.array(sorted(df[COL].unique()))

features = {}
for i, entity in enumerate(entities):
    print("Entity: {} - ".format(entity), end="")

    df_ent = df[df[COL] == entity].reset_index(drop=True)
    df_ent.drop(columns=[COL], inplace=True)
    df_ent = pd.concat([df_ent, avg_delays], axis=1)

    cols, cols_imp = u.get_feature_importance(df_ent, SELECTOR_RF, cols_data, PERM_COLS, ALT_COLS, 'MEAN_DELAY', 2, N_SAMPLES)
    features[entity] = {"cols": cols, "imp": cols_imp}
    print("DONE - Cols: {}".format(cols[len(PERM_COLS):]))

with open(OUT_PATH + OUT_FILE, 'w') as fp:
    json.dump(features, fp)
