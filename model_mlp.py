import pandas as pd
import json
import numpy as np

import utils as u

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping


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
                X_train, y_train, X_test, y_test = u.obtain_training_data(df_train, df_test, H, COL_DELAY, N_SAMPLES)

                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                                  test_size=0.2,
                                                                  random_state=42,
                                                                  shuffle=True)

                model = Sequential()
                model.add(Dense(1000, activation='relu', input_dim=X_train.shape[1]))
                model.add(Dropout(0.5))
                model.add(Dense(1000, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(y_train.shape[1], activation='sigmoid'))

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

            metrics_ent = u.get_metrics_dict(np.mean(metrics, axis=0))

            results[entity] = metrics_ent
            print("DONE - Results: {}\n".format(metrics_ent))

        with open(DATA_PATH + 'results_' + feat.lower() + '_' + col.lower(), 'w') as fp:
            json.dump(results, fp)


