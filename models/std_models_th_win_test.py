import pandas as pd
import json
import numpy as np
from multiprocessing import cpu_count, Pool

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.base import clone
from sklearn.model_selection import train_test_split

import os
from datetime import datetime
import sys
sys.path.append('..')

import utils as u


DATA_FILE_NODES = 'incoming_delays_nodes.csv'
DATA_FILE_OD_PAIRS = 'incoming_delays_ods.csv'
DATA_FILE_DELAYS = 'avg_delays.csv'
DATA_PATH = '/home/server/Aero/features/'
FEATSEL_PATH = '/home/server/Aero/modelIn/'
RESULTS_PATH = '/home/server/Aero/results/'
DATA_PATH = "C:\\Users\\victor\\Documents\\Aero_TFG\\features\\"
FEATSEL_PATH = "C:\\Users\\victor\\Documents\\Aero_TFG\\modelIn\\"
RESULTS_PATH = "C:\\Users\\victor\\Documents\\Aero_TFG\\results\\"

COLS = ['NODE', 'OD_PAIR']
THRESHOLDS = [30, 60, 120, 240]
WINDOWS = [1, 2, 4, 6]
N_SAMPLES = 3000
COL_DELAY = 'MEAN_DELAY'
COL_CLAS = 'y_clas'
H = 2
ITERS = 10

def test_entity(m, entity, df_ent, df_2, cols):
    print("Entity: {} - ".format(entity), end="")

    df = pd.concat([df_ent, df_2], axis=1)

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

        model = clone(m).fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics.append(u.get_metrics(y_test, y_pred))

        y_pred_bal = model.predict(X_test_bal)
        metrics_bal.append(u.get_metrics(y_test_bal, y_pred_bal))

        y_pred_assump = u.predict_assumption(data, model)
        metrics_assumption.append(u.get_metrics(y_test, y_pred_assump))

    metrics_ent = u.get_metrics_dict(np.mean(metrics, axis=0))

    results_ent = {
        'unbal': metrics_ent,
        'bal': u.get_metrics_dict(np.mean(metrics_bal, axis=0)),
        'assumption': u.get_metrics_dict(np.mean(metrics_assumption, axis=0))
    }
    print("DONE {} - Results: {}\n".format(entity, metrics_ent))

    return results_ent


if __name__ == '__main__':
    datestr = datetime.now().strftime("%Y%m%d-%H%M")

    results_folder = RESULTS_PATH + datestr + 'std_models_allAlt/'
    os.mkdir(results_folder)

    models = {
        'DT': DecisionTreeClassifier(max_depth=15),
        'RF': RandomForestClassifier(n_estimators=100,
                                     max_depth=5,
                                     random_state=0,
                                     criterion='gini'),
        'LR': LogisticRegression(random_state=0,
                                 solver='liblinear',
                                 penalty='l1',
                                 C=.9),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'GBT': GradientBoostingClassifier(n_estimators=100,
                                          max_depth=5,
                                          random_state=0,
                                          criterion='friedman_mse')
    }
    N_CPUS = cpu_count()
    print("N cpus: " + str(N_CPUS))

    df_nodes = pd.read_csv(DATA_PATH + DATA_FILE_NODES, sep='|').fillna(0)
    df_odpairs = pd.read_csv(DATA_PATH + DATA_FILE_OD_PAIRS, sep='|').fillna(0)
    df_2 = pd.read_csv(DATA_PATH + DATA_FILE_DELAYS, sep='|')

    feat = 'RF'

    for th in THRESHOLDS:
        for win in WINDOWS:
            for col in COLS:

                # if feat != 'ALL':
                with open(FEATSEL_PATH + 'feat_{}_{}s.json'.format(feat.lower(), col.lower()), 'r') as f:
                    features = json.load(f)

                df_1 = df_nodes if col == 'NODE' else df_odpairs

                entities = np.array(sorted(df_1[col].unique()))

                df_1 = u.get_label(df_1, th, win, entities.shape[0]).drop(columns='y_reg')

                for name, model in models.items():

                    results = {}

                    with Pool(processes=N_CPUS) as p:
                        
                        procs = {}
                        for i, entity in enumerate(entities):
                            print("Model: {} - Entity: {} - i: {}".format(name, entity, i))
                            # cols = df.columns.drop(['FL_DATE', col], errors='ignore') if feat == 'ALL' else (features[entity]['cols'] + ['y_clas'])
                            cols = features[entity]['cols'] + ['y_clas']
                            procs[entity] = p.apply_async(test_entity, args=[model,
                                                        entity, df_1[df_1[col] == entity].reset_index(drop=True),
                                                        df_2,
                                                        cols])

                        for i, entity in enumerate(entities):
                            results[entity] = procs[entity].get()
                            print("Model: {}, Entity: {}, i: {}, Results: {}".format(name, entity, i, results[entity]['unbal']))

                    with open(results_folder + 'results_' + name + '_' + feat.lower() + '_' + col.lower() + '.json', 'w') as fp:
                        json.dump(results, fp)
