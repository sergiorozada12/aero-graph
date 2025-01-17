import pandas as pd
import json
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def get_metrics_dict(metrics_arr):
    return {'acc': metrics_arr[0],
            'prec': metrics_arr[1],
            'rec': metrics_arr[2],
            'f1': metrics_arr[3]}


def get_metrics(y_test, y_pred):
    return [accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred)]


def obtain_training_data(df_train, df_test, h, col_delay, n_samples):
    df_ = df_train.loc[df_train[col_delay].shift(-h).fillna(0) != 0]

    df_delay = df_[df_['y_clas'] == 1]

    idx_delay_repeated = np.repeat(df_delay.index, (n_samples // len(df_delay)) + 1)

    idx_non_delay = np.random.choice(df_[df_['y_clas'] == 0].index, n_samples // 2, replace=False)
    idx_delay = np.random.choice(idx_delay_repeated, n_samples // 2, replace=False)
    idx = np.concatenate([idx_non_delay, idx_delay])

    X_train = df_.loc[idx, df_train.columns[:-1]].values
    y_train = df_.loc[idx, df_train.columns[-1]].values

    X_test = df_test.values[:, :-1]
    y_test = df_test.values[:, -1]

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


DATA_PATH = "data/"

DATA_FILE_NODES = 'incoming_delays_nodes.csv'
DATA_FILE_OD_PAIRS = 'incoming_delays_ods.csv'
DATA_FILE_DELAYS = 'avg_delays.csv'
FEATURES_JSON_LR = 'features_node_all_lr.json'
FEATURES_JSON_RF = 'features_node_all_lr.json'

COLS = ['NODE', 'OD_PAIR']
FEATS = ['LR', 'RF']
N_SAMPLES = 3000
COL_DELAY = 'MEAN_DELAY'
H = 2
ITERS = 10

df_nodes = pd.read_csv(DATA_PATH + DATA_FILE_NODES, sep='|').fillna(0)
df_odpairs = pd.read_csv(DATA_PATH + DATA_FILE_OD_PAIRS, sep='|').fillna(0)
df_2 = pd.read_csv(DATA_PATH + DATA_FILE_DELAYS, sep='|')

with open(DATA_PATH + FEATURES_JSON_LR, 'r') as f:
    features_lr = json.load(f)

with open(DATA_PATH + FEATURES_JSON_RF, 'r') as f:
    features_rf = json.load(f)

for feat in FEATS:
    features = features_rf if feat == 'RF' else features_lr

    for col in COLS:

        if col == 'NODE':
            df_1 = df_nodes
            entities = np.array(sorted(df_1[col].unique()))
        else:
            df_1 = df_odpairs
            entities = np.array(sorted(df_1[col].unique()))

        results = {}
        for i, entity in enumerate(entities):
            print("Entity: {} - ".format(entity), end="")

            df_ent = df_1[df_1[col] == entity].reset_index(drop=True)
            df = pd.concat([df_ent, df_2], axis=1)

            df_train = df[df['YEAR'] == 2018][features[entity]['cols'] + ['y_clas']].reset_index(drop=True)
            df_test = df[df['YEAR'] == 2019][features[entity]['cols'] + ['y_clas']].reset_index(drop=True)

            metrics = []
            for j in range(ITERS):
                X_train, y_train, X_test, y_test = obtain_training_data(df_train, df_test, H, COL_DELAY, N_SAMPLES)

                model = LogisticRegression(random_state=0,
                                           solver='liblinear',
                                           penalty='l1',
                                           C=.9).fit(X_train, y_train)

                y_pred = model.predict(X_test)
                metrics.append(get_metrics(y_test, y_pred))

            metrics_ent = get_metrics_dict(np.mean(metrics, axis=0))

            results[entity] = metrics_ent
            print("DONE - Results: {}\n".format(metrics_ent))

        with open(DATA_PATH + 'results_' + feat.lower() + '_' + col.lower(), 'w') as fp:
            json.dump(results, fp)

