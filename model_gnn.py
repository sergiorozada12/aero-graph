import pandas as pd
import numpy as np
import json

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from gnn.model import Model, ADAM
from gnn.arch import BasicArch

import utils as u


DATA_PATH = '/home/server/Aero/features/'
FEATSEL_PATH = '/home/server/Aero/modelIn/'
GRAPH_PATH = '/home/server/Aero/data/graph/'
RESULTS_PATH = '/home/server/Aero/results/'

VERB = True
ARCH_INFO = True

DELAY_TYPES = ['CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']


def eval_arch(X_train, y_train, mlp_features_train, X_val, y_val, mlp_features_val, X_test, y_test, mlp_features_test, X_unbal, y_unbal, mlp_features_unbal):
    archit = BasicArch(**arch_params)
    model_params['arch'] = archit

    print("Started Training")

    model = Model(**model_params)
    epochs, train_err, val_err = model.fit(X_train, y_train, mlp_features_train, X_val, y_val, mlp_features_val)

    mean_train_err = np.mean(train_err)
    mean_val_err = np.mean(val_err)

    print("Finished Training Model")
    print("Epochs: {} - Mean Train Error: {} - Mean Validation Error: {}".format(
        epochs, mean_train_err, mean_val_err
    ))

    print("Started Testing")

    # test_loss, accuracy, precision, recall = model.test(X_test, y_test, mlp_features_test)
    # loss_unbal, acc_unbal, prec_unbal, rec_unbal = model.test(X_unbal, y_unbal, mlp_features_bal)
    # dummy_accuracy = accuracy_score(y_unbal, torch.zeros(y_unbal.size()))

    # print("----- END -------")
    # print("Test Loss: {} - Accuracy: {} - Precision: {} - Recall: {}".format(
    #     test_loss, accuracy, precision, recall
    # ))
    # print("Unbalanced - Loss: {} - Accuracy: {} - Precision: {} - Recall: {} \
    #     - Dummy Acc: {}".format(
    #     loss_unbal, acc_unbal, prec_unbal, rec_unbal, dummy_accuracy
    # ))

    y_pred_bal = model.predict(X_test, mlp_features_test)
    metrics_bal = u.get_metrics(y_test, y_pred_bal)

    y_pred_unbal = model.predict(X_unbal, mlp_features_unbal)
    metrics_unbal = u.get_metrics(y_unbal, y_pred_unbal)

    print("----- END -------")
    print("Balanced: Accuracy: {} - Precision: {} - Recall: {} - F1: {}".format(
        metrics_bal[0], metrics_bal[1], metrics_bal[2], metrics_bal[3]
    ))
    print("Unbalanced: Accuracy: {} - Precision: {} - Recall: {} - F1: {}".format(
        metrics_unbal[0], metrics_unbal[1], metrics_unbal[2], metrics_unbal[3]
    ))

    results = {}
    results['bal'] = u.get_metrics_dict(metrics_bal)
    results['unbal'] = u.get_metrics_dict(metrics_unbal)
    # results = {
    #     'epochs': epochs,
    #     'train_err': mean_train_err,
    #     'val_err': mean_val_err,
    #     'test_loss': np.mean(test_loss),
    #     'accuracy': accuracy,
    #     'precision': precision,
    #     'recall': recall,
    #     'loss_unbal': loss_unbal,
    #     'acc_unbal': acc_unbal,
    #     'prec_unbal': prec_unbal,
    #     'rec_unbal': rec_unbal
    # }

    return results

def obtain_data(df, df_d_types, avg_delays, entities):
    cols_depart = [ent + '_DEP' for ent in entities]
    cols_arriv = [ent + '_ARR' for ent in entities]

    # Data needs to be [TxFxN]
    # T is the number of samples, in this case, the number of hours
    # F is the number of features
    # N is the signal lenght

    features = []
    mlp_features = []

    for c in PERM_COLS:
        mlp_features.append(df[c].values.reshape(-1, entities.shape[0])[:,0])

    mlp_features.append(avg_delays['HOUR_CLUSTER'].values)
    mlp_features.append(avg_delays['DAY_CLUSTER'].values)
    mlp_features.append(avg_delays['DAY_CLUSTER-1'].values)

    features.append(avg_delays[cols_depart].values)
    features.append(avg_delays[cols_arriv].values)

    # Delay -1 hour
    features.append(avg_delays.shift(1)[cols_depart].fillna(0).values)
    features.append(avg_delays.shift(1)[cols_arriv].fillna(0).values)

    # Delay -2 hour
    features.append(avg_delays.shift(2)[cols_depart].fillna(0).values)
    features.append(avg_delays.shift(2)[cols_arriv].fillna(0).values)

    # Delay -1 day
    features.append(avg_delays.shift(24)[cols_depart].fillna(0).values)
    features.append(avg_delays.shift(24)[cols_arriv].fillna(0).values)

    # Delay types
    for d in DELAY_TYPES:
        cols_d = [ent + d for ent in entities]
        features.append(df_d_types[cols_d].values)

    return np.stack(features, axis=1), np.stack(mlp_features, axis=1)

# Load the signal
df = pd.read_csv(DATA_PATH + 'incoming_delays_nodes.csv', sep='|')

df_d_types = pd.read_csv(DATA_PATH + 'df_d_types.csv', sep='|')

avg_delays = pd.read_csv(DATA_PATH + 'avg_delays.csv', sep='|')

# Load the graph
S = np.load(GRAPH_PATH + 'Adj_nodes.npy', allow_pickle=True)
S = S/np.sum(S)
print(S)

PERM_COLS = ['HOUR',
             'DAY',
             'DAY_OF_WEEK',
             'MONTH',
             'QUARTER',
             'SEASON']

COL = 'NODE'
N_SAMPLES = 3000

entities = np.array(sorted(df[COL].unique()))

# Obtain data
X, mlp_features = obtain_data(df, df_d_types, avg_delays, entities)
n_feats = X.shape[1]

# Architecture parameters
arch_params = {}
arch_params['S'] = S
arch_params['F'] = [n_feats, 32, 16, 8]
#arch_params['F'] = []
arch_params['K'] = 3
arch_params['M'] = [1024, 512, 64, 32, 2]
arch_params['nonlin'] = nn.ReLU
arch_params['nonlin_mlp'] = nn.ReLU
arch_params['arch_info'] = ARCH_INFO
arch_params['n_mlp_feat'] = mlp_features.shape[1]
arch_params['dropout_mlp'] = 0.5

# Model parameters
model_params = {}
model_params['opt'] = ADAM
model_params['learning_rate'] = 0.001
model_params['decay_rate'] = 0.99
model_params['loss_func'] = nn.CrossEntropyLoss()
model_params['epochs'] = 200
model_params['batch_size'] = 10
model_params['eval_freq'] = 4
model_params['max_non_dec'] = 10
model_params['verbose'] = VERB

df_work = df[df['YEAR'] == 2018].reset_index(drop=True)
idx_work = len(df_work) // entities.shape[0]
print(len(df_work) / entities.shape[0])
X_work = X[:idx_work,:,:]
mlp_features_work = mlp_features[:idx_work,:]

df_test = df[df['YEAR'] == 2019].reset_index(drop=True)
X_test_unbal = X[idx_work:,:,:]
mlp_features_test_unbal = mlp_features[idx_work:,:]

assert (len(df_test) // entities.shape[0]) == X_test_unbal.shape[0]

results = {}
for ent in entities:
    print("Entity: {} - ".format(ent))
    df_ent = df_work[df_work[COL] == ent].reset_index(drop=True)

    idx = u.obtain_equal_idx(df_ent[df_ent['y_clas'] == 0].index, df_ent[df_ent['y_clas'] == 1].index, N_SAMPLES)

    X_unbal = X[idx, :, :]
    mlp_features_ent = mlp_features[idx,:]
    y = df_ent.loc[idx, 'y_clas'].values

    X_train, X_test_bal, y_train, y_test_bal, mlp_features_train, mlp_features_test =\
        train_test_split(X_unbal, y, mlp_features_ent, test_size=0.1)
    X_train, X_val, y_train, y_val, mlp_features_train, mlp_features_val =\
        train_test_split(X_train, y_train, mlp_features_train, test_size=0.12)

    y_unbal = df_test.loc[df_test[COL] == ent, 'y_clas'].values

    results[ent] = eval_arch(torch.Tensor(X_train),
                             torch.LongTensor(y_train),
                             torch.Tensor(mlp_features_train),
                             torch.Tensor(X_val),
                             torch.LongTensor(y_val),
                             torch.Tensor(mlp_features_val),
                             torch.Tensor(X_test_bal),
                             torch.LongTensor(y_test_bal),
                             torch.Tensor(mlp_features_test),
                             torch.Tensor(X_test_unbal),
                             torch.LongTensor(y_unbal),
                             torch.Tensor(mlp_features_test_unbal))
    #print("DONE - Test Loss: {} - Accuracy: {}".format(results[ent]['test_loss'], results[ent]['accuracy']))

with open(RESULTS_PATH + '20200319-nodes-gnn.json', 'w') as f:
    json.dump(results, f)
