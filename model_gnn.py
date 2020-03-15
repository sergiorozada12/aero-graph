import pandas as pd
import numpy as np
import json

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from gnn.model import Model, ADAM
from gnn.arch import BasicArch


DATA_PATH = '/home/server/Aero/features/'
FEATSEL_PATH = '/home/server/Aero/modelIn/'
GRAPH_PATH = '/home/server/Aero/data/graph/'
RESULTS_PATH = '/home/server/Aero/results/'

VERB = True
ARCH_INFO = True

DELAY_TYPES = ['CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']


def eval_arch(X_train, y_train, X_val, y_val, X_test, y_test, X_unbal, y_unbal):
    archit = BasicArch(**arch_params)
    model_params['arch'] = archit

    print("Started Training")

    model = Model(**model_params)
    epochs, train_err, val_err = model.fit(X_train, y_train, X_val, y_val)

    mean_train_err = np.mean(train_err)
    mean_val_err = np.mean(val_err)

    print("Finished Training Model")
    print("Epochs: {} - Mean Train Error: {} - Mean Validation Error: {}".format(
        epochs, mean_train_err, mean_val_err
    ))

    print("Started Testing")

    test_loss, accuracy, precision, recall = model.test(X_test, y_test)
    loss_unbal, acc_unbal, prec_unbal, rec_unbal = model.test(X_unbal, y_unbal)
    dummy_accuracy = accuracy_score(y_test, torch.zeros(y_test.size()))

    print("----- END -------")
    print("Test Loss: {} - Accuracy: {} - Precision: {} - Recall: {} \
            - Dummy Acc: {} - Accuracy Unbalanced: {}".format(
        test_loss, accuracy, precision, recall, dummy_accuracy, acc_unbal
    ))

    results = {
        'epochs': epochs,
        'train_err': train_err,
        'val_err': val_err,
        'test_loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'loss_unbal': loss_unbal,
        'acc_unbal': acc_unbal,
        'prec_unbal': prec_unbal,
        'rec_unbal': rec_unbal
    }

    return results

def obtain_data(df_d_types, avg_delays, entities):
    cols_depart = [ent + '_DEP' for ent in entities]
    cols_arriv = [ent + '_ARR' for ent in entities]

    # Data needs to be [TxFxN]
    # T is the number of samples, in this case, the number of hours
    # F is the number of features
    # N is the signal lenght

    features = []

    # for c in PERM_COLS:
    #     features.append(torch.Tensor(df[c].values.reshape(-1, entities.shape[0])))

    features.append(avg_delays[cols_depart].values)
    features.append(avg_delays[cols_arriv].values)

    # Delay -1 hour
    features.append(avg_delays.shift(1)[cols_arriv].fillna(0).values)

    # Delay -2 hour
    features.append(avg_delays.shift(2)[cols_arriv].fillna(0).values)

    # Delay -1 day
    features.append(avg_delays.shift(24)[cols_arriv].fillna(0).values)

    # Delay types
    for d in DELAY_TYPES:
        cols_d = [ent + d for ent in entities]
        features.append(df_d_types[cols_d].values)

    return np.stack(features, axis=1)

# Load the signal
df = pd.read_csv(DATA_PATH + 'incoming_delays_nodes.csv', sep='|')

df_d_types = pd.read_csv(DATA_PATH + 'df_d_types.csv', sep='|')

avg_delays = pd.read_csv(DATA_PATH + 'avg_delays.csv', sep='|')

# Load the graph
S = np.load(GRAPH_PATH + 'Adj_nodes.npy', allow_pickle=True)

PERM_COLS = ['HOUR',
             'DAY',
             'DAY_OF_WEEK',
             'MONTH',
             'QUARTER',
             'SEASON']

COL = 'NODE'
N_SAMPLES = 5000

entities = np.array(sorted(df[COL].unique()))

# Obtain data
X = obtain_data(df_d_types, avg_delays, entities)
n_feats = X.size()[1]

# Architecture parameters
arch_params = {}
arch_params['S'] = S
arch_params['F'] = [n_feats, 32, 16, 8]
arch_params['K'] = 3
arch_params['M'] = [32, 2]
arch_params['nonlin'] = nn.Tanh
arch_params['arch_info'] = ARCH_INFO

# Model parameters
model_params = {}
model_params['opt'] = ADAM
model_params['learning_rate'] = 0.001
model_params['decay_rate'] = 0.99
model_params['loss_func'] = nn.CrossEntropyLoss()
model_params['epochs'] = 200
model_params['batch_size'] = 50
model_params['eval_freq'] = 4
model_params['max_non_dec'] = 10
model_params['verbose'] = VERB

results = {}
for ent in entities:
    print("Entity: {} - ".format(ent))
    df_ent = df[df[COL] == ent].reset_index(drop=True)

    n_delays = (df_ent['y_clas'] == 1).sum()

    idx_delay_repeated = np.repeat(df_ent[df_ent['y_clas'] == 1].index, (N_SAMPLES // n_delays) + 1)
    idx_non_delay = np.random.choice(df_ent[df_ent['y_clas'] == 0].index, N_SAMPLES // 2, replace=False)
    idx_delay = np.random.choice(idx_delay_repeated, N_SAMPLES // 2, replace=False)
    idx = np.concatenate([idx_non_delay, idx_delay])

    X_unbal = X[idx, :, :]
    y = df_ent.loc[idx, 'y_clas'].values

    X_train, X_test, y_train, y_test = train_test_split(X_unbal, y, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.12)

    y_bal = df_ent['y_clas'].values
    
    results[ent] = eval_arch(torch.Tensor(X_train),
                             torch.LongTensor(y_train),
                             torch.Tensor(X_val),
                             torch.LongTensor(y_val),
                             torch.Tensor(X_test),
                             torch.LongTensor(y_test),
                             torch.Tensor(X),
                             torch.LongTensor(y_bal))
    #print("DONE - Test Loss: {} - Accuracy: {}".format(results[ent]['test_loss'], results[ent]['accuracy']))