import pandas as pd
import json
import numpy as np

import utils as u

from sklearn.model_selection import train_test_split

from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping


DATA_FILE_NODES = 'incoming_delays_nodes.csv'
DATA_FILE_OD_PAIRS = 'incoming_delays_ods.csv'
DATA_FILE_DELAYS = 'avg_delays.csv'
DATA_PATH = '/home/server/Aero/features/'
FEATSEL_PATH = '/home/server/Aero/modelIn/'
RESULTS_PATH = '/home/server/Aero/results/'

COLS = ['NODE', 'OD_PAIR']
FEATS = ['LR', 'RF']
N_SAMPLES = 4000
COL_DELAY = 'MEAN_DELAY'
H = 2
ITERS = 10
N_STEPS = 3

df_nodes = pd.read_csv(DATA_PATH + DATA_FILE_NODES, sep='|').fillna(0)
df_odpairs = pd.read_csv(DATA_PATH + DATA_FILE_OD_PAIRS, sep='|').fillna(0)
df_2 = pd.read_csv(DATA_PATH + DATA_FILE_DELAYS, sep='|')

for feat in FEATS:

    for col in COLS:

        with open(FEATSEL_PATH + 'feat_{}_{}s.json'.format(feat.lower(), col.lower()), 'r') as f:
            features = json.load(f)

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
            df_ = pd.concat([df_ent, df_2], axis=1)

            columns = df_.columns
            df = df_.copy()

            for c in columns:
                df[c] = df[c].astype('object')

            for idx, row in df_.iterrows():
                for c in columns:
                    if c != 'y_clas':
                        df.at[idx, c] = df_.loc[idx - N_STEPS + 1:idx, c].tolist()

            df = df.drop(df.index[0:N_STEPS - 1]).reset_index(drop=True)

            df_train = df[df['YEAR'] == 2018][features[entity]['cols'] + ['y_clas']].reset_index(drop=True)
            df_test = df[df['YEAR'] == 2019][features[entity]['cols'] + ['y_clas']].reset_index(drop=True)

            n_cols_X = len(features[entity]['cols'])

            metrics = []
            metrics_bal = []
            metrics_0s = []
            for j in range(ITERS):
                X_0s, y_0s, X_train, y_train, X_test, y_test =\
                    u.obtain_training_data(df_train, df_test, H, COL_DELAY, N_SAMPLES)
                X_train, X_no0s_bal, y_train, y_no0s_bal = train_test_split(X_train, y_train, test_size=0.15)

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
                                    shuffle=True)

                y_pred = (model.predict(X_test) >= .5) * 1
                metrics.append(u.get_metrics(y_test, y_pred))

                y_pred_bal = (model.predict(X_no0s_bal) >= .5) * 1
                metrics_bal.append(u.get_metrics(y_no0s_bal, y_pred_bal))

                y_pred_0s = (model.predict(X_0s) >= .5) * 1
                metrics_0s.append(u.get_metrics(y_0s, y_pred_0s))

            metrics_ent = u.get_metrics_dict(np.mean(metrics, axis=0))

            results[entity] = {
                'unbal': metrics_ent,
                'bal': u.get_metrics_dict(np.mean(metrics_bal, axis=0)),
                'bal_zeros': u.get_metrics_dict(np.mean(metrics_0s, axis=0))
            }
            print("DONE - Results: {}\n".format(metrics_ent))

        with open(RESULTS_PATH + 'results_MLP_' + feat.lower() + '_' + col.lower() + '.json', 'w') as fp:
            json.dump(results, fp)


