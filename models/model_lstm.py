import pandas as pd
import json
import numpy as np
from datetime import datetime

import sys
sys.path.append('..')

import utils as u

from sklearn.model_selection import train_test_split

from keras.layers import LSTM
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
FEATS = ['RF']
N_SAMPLES = 4000
COL_DELAY = 'MEAN_DELAY'
COL_CLAS = 'y_clas'
H = 2
ITERS = 10
N_STEPS = 3
N_COLS = 20

df_nodes = pd.read_csv(DATA_PATH + DATA_FILE_NODES, sep='|').fillna(0)
df_odpairs = pd.read_csv(DATA_PATH + DATA_FILE_OD_PAIRS, sep='|').fillna(0)
df_2 = pd.read_csv(DATA_PATH + DATA_FILE_DELAYS, sep='|')

def now_str():
    return datetime.now().strftime("%Y%m%d-%H:%M:%S")

for feat in FEATS:

    for col in COLS:

        if feat != 'ALL':
            with open(FEATSEL_PATH + 'feat_{}_{}s.json'.format(feat.lower(), col.lower()), 'r') as f:
                features = json.load(f)

        df_1 = df_nodes if col == 'NODE' else df_odpairs
        
        entities = np.array(sorted(df_1[col].unique()))

        results = {}
        for i, entity in enumerate(entities):
            print("{} - Entity: {}".format(now_str(), entity))

            df_ent = df_1[df_1[col] == entity].reset_index(drop=True)
            df_ = pd.concat([df_ent, df_2], axis=1)

            cols = features[entity]['cols'][:N_COLS] + [COL_CLAS]
            if COL_DELAY not in cols:
                cols[-2] = COL_DELAY

            df = df_.copy()[cols]

            for c in cols:
                if c != COL_CLAS:
                    df[c] = df[c].astype('object')

            for idx, row in df_.iterrows():
                for c in cols:
                    if c != COL_CLAS:
                        df.at[idx, c] = df_.loc[idx - N_STEPS + 1:idx, c].tolist()

            df = df.drop(df.index[0:N_STEPS - 1]).reset_index(drop=True)

            df['YEAR_IND'] = df_['YEAR'].copy()

            df_train = df[df['YEAR_IND'] == 2018].reset_index(drop=True).drop(columns='YEAR_IND')
            df_test = df[df['YEAR_IND'] == 2019].reset_index(drop=True).drop(columns='YEAR_IND')

            n_cols_X = N_COLS

            data = u.TrainingData(df_train, df_test, H, COL_DELAY, COL_CLAS)

            metrics = []
            metrics_bal = []
            #metrics_assumption = []
            print("{} - Starting training".format(now_str()))
            for j in range(ITERS):
                X_train, y_train, X_test, y_test =\
                    data.obtain_training_data(N_SAMPLES, lstm=True)
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
                model.add(LSTM(1000, input_shape=(N_STEPS, n_cols_X)))
                model.add(Dropout(0.5))
                model.add(Dense(1000, activation='relu'))
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

                # y_pred_assump = (u.predict_assumption(data, model) >= .5) * 1
                # metrics_assumption.append(u.get_metrics(y_test, y_pred_assump))

                clear_session()

            metrics_ent = u.get_metrics_dict(np.mean(metrics, axis=0))

            results[entity] = {
                'unbal': metrics_ent,
                'bal': u.get_metrics_dict(np.mean(metrics_bal, axis=0)),
                #'bal_zeros': u.get_metrics_dict(np.mean(metrics_assumption, axis=0))
            }
            print("{} - DONE - Results: {}\n".format(now_str(), metrics_ent))

        with open(RESULTS_PATH + 'results_LSTM_' + feat.lower() + '_' + col.lower() + '.json', 'w') as fp:
            json.dump(results, fp)


