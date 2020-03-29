import pandas as pd
import json
import numpy as np

import sys
sys.path.append('..')

import utils as u

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.backend import clear_session


DATA_FILE_NODES = 'incoming_delays_nodes.csv'
DATA_FILE_OD_PAIRS = 'incoming_delays_ods.csv'
DATA_FILE_DELAYS = 'avg_delays.csv'
DATA_PATH = '/home/server/Aero/features/'
FEATSEL_PATH = '/home/server/Aero/modelIn/'
RESULTS_PATH = '/home/server/Aero/results/'

COLS = ['OD_PAIR', 'NODE']
FEATS = ['ALL', 'LR', 'RF']
N_SAMPLES = 4000
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
            cols = df.columns.drop(['FL_DATE', col], errors='ignore') if feat == 'ALL' else (features[entity]['cols'] + ['y_clas'])

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

                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                                  test_size=0.2,
                                                                  random_state=42,
                                                                  shuffle=True)

                try:
                    features_out = y_train.shape[1]
                except IndexError:
                    features_out = 1

                model = Sequential()

                model.add(Dense(1000, activation='relu', input_dim=X_train.shape[1]))
                model.add(Dropout(0.5))
                model.add(Dense(1000, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(features_out, activation='sigmoid'))

                model.compile(loss='binary_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

                es = EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   verbose=1,
                                   patience=10)

                history = model.fit(X_train,
                                    y_train,
                                    validation_data=(X_val, y_val),
                                    epochs=100,
                                    batch_size=128,
                                    callbacks=[es],
                                    shuffle=True,
                                    verbose=0)

                y_pred = (model.predict(X_test) >= .5) * 1
                metrics.append(u.get_metrics(y_test, y_pred))

                y_pred_bal = (model.predict(X_test_bal) >= .5) * 1
                metrics_bal.append(u.get_metrics(y_test_bal, y_pred_bal))

                y_pred_assump = (u.predict_assumption(data, model) >= .5) * 1
                metrics_assumption.append(u.get_metrics(y_test, y_pred_assump))

                clear_session()

            metrics_ent = u.get_metrics_dict(np.mean(metrics, axis=0))

            results[entity] = {
                'unbal': metrics_ent,
                'bal': u.get_metrics_dict(np.mean(metrics_bal, axis=0)),
                'assumption': u.get_metrics_dict(np.mean(metrics_assumption, axis=0))
            }
            print("DONE - Results: {}\n".format(metrics_ent))

        with open(RESULTS_PATH + 'results_MLP_' + feat.lower() + '_' + col.lower() + '.json', 'w') as fp:
            json.dump(results, fp)


