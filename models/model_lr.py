import pandas as pd
import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')

import utils as u


DATA_FILE_NODES = 'incoming_delays_nodes.csv'
DATA_FILE_OD_PAIRS = 'incoming_delays_ods.csv'
DATA_FILE_DELAYS = 'avg_delays.csv'
DATA_PATH = '/home/server/Aero/features/'
FEATSEL_PATH = '/home/server/Aero/modelIn/'
RESULTS_PATH = '/home/server/Aero/results/'

COLS = ['NODE', 'OD_PAIR']
FEATS = ['ALL', 'LR', 'RF']
N_SAMPLES = 3000
COL_DELAY = 'MEAN_DELAY'
COL_CLAS = 'y_clas'
H = 2
ITERS = 10

df_nodes = pd.read_csv(DATA_PATH + DATA_FILE_NODES, sep='|').fillna(0)
df_odpairs = pd.read_csv(DATA_PATH + DATA_FILE_OD_PAIRS, sep='|').fillna(0)
df_2 = pd.read_csv(DATA_PATH + DATA_FILE_DELAYS, sep='|')

for feat in FEATS:

    for col in COLS:

        if feat != 'ALL':
            with open(FEATSEL_PATH + 'feat_{}_{}s.json'.format(feat.lower(), col.lower()), 'r') as f:
                features = json.load(f)

        df_1 = df_nodes if col == 'NODE' else df_odpairs
        
        entities = np.array(sorted(df_1[col].unique()))

        results = {}
        for i, entity in enumerate(entities):
            print("Entity: {} - ".format(entity), end="")

            df_ent = df_1[df_1[col] == entity].reset_index(drop=True)
            df = pd.concat([df_ent, df_2], axis=1)

            cols = df.columns.drop(['FL_DATE', col]) if feat == 'ALL' else (features[entity]['cols'] + ['y_clas'])

            df_train = df[df['YEAR'] == 2018][cols].reset_index(drop=True)
            df_test = df[df['YEAR'] == 2019][cols].reset_index(drop=True)

            data = u.TrainingData(df_train, df_test, H, COL_DELAY, COL_CLAS)

            metrics = []
            metrics_bal = []
            metrics_assumption = []
            for j in range(ITERS):
                X_train, y_train, X_test, y_test =\
                    data.obtain_training_data(N_SAMPLES)
                X_train, X_test_bal, y_train, y_test_bal = train_test_split(X_train, y_train, test_size=0.15)

                model = LogisticRegression(random_state=0,
                                           solver='liblinear',
                                           penalty='l1',
                                           C=.9).fit(X_train, y_train)

                y_pred = model.predict(X_test)
                metrics.append(u.get_metrics(y_test, y_pred))

                y_pred_bal = model.predict(X_test_bal)
                metrics_bal.append(u.get_metrics(y_test_bal, y_pred_bal))

                y_pred_assump = u.predict_assumption(data, model)
                metrics_assumption.append(u.get_metrics(y_test, y_pred_assump))

            metrics_ent = u.get_metrics_dict(np.mean(metrics, axis=0))

            results[entity] = {
                'unbal': metrics_ent,
                'bal': u.get_metrics_dict(np.mean(metrics_bal, axis=0)),
                'assumption': u.get_metrics_dict(np.mean(metrics_assumption, axis=0))
            }
            print("DONE - Results: {}\n".format(metrics_ent))

        with open(RESULTS_PATH + 'results_LR_' + feat.lower() + '_' + col.lower() + '.json', 'w') as fp:
            json.dump(results, fp)
