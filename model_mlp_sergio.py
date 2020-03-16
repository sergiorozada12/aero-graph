import pandas as pd
import json
import numpy as np

import utils as u

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping

def undersample(df, ratio_imbalance, col_target):

    random_indices = np.random.choice(np.where(df[col_target] == 0)[0],
                                      ratio_imbalance*np.sum(df[col_target] == 1),
                                      replace=False)

    idx = np.concatenate([random_indices, np.where(df[col_target] == 1)[0]])

    return df.loc[idx, :].reset_index(drop=True)

DATA_PATH = '/home/server/Aero/features/'
RESULTS_PATH = '/home/server/Aero/results/'

DATA_FILE = 'incoming_delays_nodes.csv'
RESULTS_JSON = 'results_model_sergio.json'

COL = 'NODE'
RATIO_IMB = 3

COLS_TRAIN = ['HOUR',
              'DAY',
              'DAY_OF_WEEK',
              'MONTH',
              'QUARTER',
              'YEAR',
              'MEAN_DELAY',
              'INCOMING_MEAN_DELAY',
              'MEAN_CARRIER_DELAY',
              'MEAN_LATE_AIRCRAFT_DELAY',
              'MEAN_NAS_DELAY',
              'MEAN_SECURITY_DELAY',
              'MEAN_WEATHER_DELAY',
              'y_clas']

COL_TARGET = COLS_TRAIN[-1]

df = pd.read_csv(DATA_PATH + DATA_FILE, sep='|').fillna(0)

entities = np.array(sorted(df[COL].unique()))

results = {}
y_test_acc = np.array([])
y_pred_acc = np.array([])
for i, entity in enumerate(entities):
    print("Entity: {} - ".format(entity), end="")

    df_ent = df[df[COL] == entity].reset_index(drop=True)[COLS_TRAIN]

    df_train = df_ent[df_ent['YEAR'] == 2018].reset_index(drop=True)
    df_test = df_ent[df_ent['YEAR'] == 2019].reset_index(drop=True)

    df_train_undersampled = undersample(df_train, RATIO_IMB, COL_TARGET)

    X_train = df_train_undersampled[COLS_TRAIN[:-1]].values
    X_test = df_test[COLS_TRAIN[:-1]].values
    y_train = df_train_undersampled[COL_TARGET].values
    y_test = df_test[COL_TARGET].values

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      shuffle=True)

    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

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

    y_pred = (model.predict(X_test) >= .5)*1
    metrics_ent = u.get_metrics_dict(u.get_metrics(y_test, y_pred))

    y_pred_val = (model.predict(X_val) >= .5) * 1

    results[entity] = {
        'unbal': metrics_ent,
        'bal': u.get_metrics_dict(u.get_metrics(y_val, y_pred_val))
    }
    print("DONE - Results: {}\n".format(metrics_ent))

    y_test_acc = np.concatenate([y_test_acc, y_test])
    y_pred_acc = np.concatenate([y_pred_acc, y_pred.flatten()])

    metrics_acc = u.get_metrics_dict(u.get_metrics(y_test_acc, y_pred_acc))
    print("Accumulated results: {}\n".format(metrics_acc))

results['TOTAL'] = metrics_acc

with open(RESULTS_PATH + RESULTS_JSON, 'w') as fp:
    json.dump(results, fp)

