import pandas as pd
import numpy as np
import json
from multiprocessing import cpu_count, Pool

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append('..')

from gnn.model import Model, ADAM
from gnn.arch import BasicArch

import utils as u


DATA_FILE_NODES = 'incoming_delays_nodes.csv'
DATA_FILE_OD_PAIRS = 'incoming_delays_ods.csv'
DATA_FILE_DELAYS = 'avg_delays.csv'
DATA_FILE_D_TYPES = 'df_d_types.csv'
DATA_PATH = '/home/server/Aero/features/'
FEATSEL_PATH = '/home/server/Aero/modelIn/'
GRAPH_PATH = '/home/server/Aero/data/graph/'
RESULTS_PATH = '/home/server/Aero/results/'

VERB = False
ARCH_INFO = False

PERM_COLS = ['HOUR',
            'DAY',
            'DAY_OF_WEEK',
            'MONTH',
            'QUARTER',
            'SEASON']

COLS = ['OD_PAIR', 'NODE']
N_SAMPLES = 3000
COL_DELAY = 'MEAN_DELAY'
COL_CLAS = 'y_clas'
H = 2
ITERS = 8

DELAY_TYPES = ['CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']


def eval_arch(X_train, y_train, mlp_features_train,
              X_val, y_val, mlp_features_val,
              X_test, y_test, mlp_features_test,
              X_unbal, y_unbal, mlp_features_unbal,
              idx_p0):
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

    X_assumption = X_unbal[idx_p0,:,:]
    mlp_features_assumption = mlp_features_unbal[idx_p0,:]
    preds_assumption = np.zeros(X_unbal.shape[0])
    y_pred_assumption = model.predict(X_assumption, mlp_features_assumption)
    preds_assumption[idx_p0] = y_pred_assumption
    metrics_assumption = u.get_metrics(y_unbal, preds_assumption)

    print("----- END -------")
    print("Balanced: Accuracy: {} - Precision: {} - Recall: {} - F1: {}".format(
        metrics_bal[0], metrics_bal[1], metrics_bal[2], metrics_bal[3]
    ))
    print("Unbalanced: Accuracy: {} - Precision: {} - Recall: {} - F1: {}".format(
        metrics_unbal[0], metrics_unbal[1], metrics_unbal[2], metrics_unbal[3]
    ))
    print("Assumption: Accuracy: {} - Precision: {} - Recall: {} - F1: {}".format(
        metrics_assumption[0], metrics_assumption[1], metrics_assumption[2], metrics_assumption[3]
    ))

    # results = {}
    # results['bal'] = u.get_metrics_dict(metrics_bal)
    # results['unbal'] = u.get_metrics_dict(metrics_unbal)
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

    return metrics_bal, metrics_unbal, metrics_assumption

def test_entity(df_ent, df_test_ent, mlp_features, X_test_unbal, mlp_features_test_unbal):

    idx_p0 = df_test_ent.loc[df_test_ent[COL_DELAY] > 0].index

    metrics_bal = np.zeros((ITERS, 4))
    metrics_unbal = np.zeros((ITERS, 4))
    metrics_assumption = np.zeros((ITERS, 4))

    for i in range(ITERS):
        
        df_ = df_ent.loc[df_ent[COL_DELAY].shift(-H).fillna(0) != 0]
        idx = obtain_equal_idx(df_[df_['y_clas'] == 0].index, df_[df_['y_clas'] == 1].index, N_SAMPLES)

        X_bal = X[idx, :, :]
        mlp_features_ent = mlp_features[idx,:]
        y = df_ent.loc[idx, 'y_clas'].values

        X_train, X_test_bal, y_train, y_test_bal, mlp_features_train, mlp_features_test =\
            train_test_split(X_bal, y, mlp_features_ent, test_size=0.1, shuffle=True)
        X_train, X_val, y_train, y_val, mlp_features_train, mlp_features_val =\
            train_test_split(X_train, y_train, mlp_features_train, test_size=0.12, shuffle=True)

        # scaler = MinMaxScaler()
        # scaler.fit(X_train)

        # X_train = scaler.transform(X_train)
        # X_test_bal = scaler.transform(X_test_bal)
        # X_val = scaler.transform(X_val)
        # X_test_unbal_it = scaler.transform(X_test_unbal)

        # scaler_mlp_features = MinMaxScaler()
        # scaler_mlp_features.fit(mlp_features_train)

        # mlp_features_train = scaler_mlp_features.transform(mlp_features_train)
        # mlp_features_val = scaler_mlp_features.transform(mlp_features_val)
        # mlp_features_test = scaler_mlp_features.transform(mlp_features_test)
        # mlp_features_test_unbal_it = scaler_mlp_features.transform(mlp_features_test_unbal)

        y_unbal = df_test_ent['y_clas'].values

        metrics_bal[i,:], metrics_unbal[i,:], metrics_assumption[i,:] = eval_arch(torch.Tensor(X_train),
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
                                                                                    torch.Tensor(mlp_features_test_unbal),
                                                                                    idx_p0)
    metrics_ent = u.get_metrics_dict(np.mean(metrics_unbal, axis=0))

    results_ent = {
        'unbal': metrics_ent,
        'bal': u.get_metrics_dict(np.mean(metrics_bal, axis=0)),
        'assumption': u.get_metrics_dict(np.mean(metrics_assumption, axis=0))
    }

    return results_ent

def obtain_equal_idx(idx_0, idx_1, n_samples):
    idx_1_repeated = np.repeat(idx_1, (n_samples // len(idx_1)) + 1)

    idx_non_delay = np.random.choice(idx_0, n_samples // 2, replace=False)
    idx_delay = np.random.choice(idx_1_repeated, n_samples // 2, replace=False)
    return np.concatenate([idx_non_delay, idx_delay])


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

if __name__ == '__main__':

    N_CPUS = cpu_count()

    # Load the signal
    df_nodes = pd.read_csv(DATA_PATH + DATA_FILE_NODES, sep='|')

    df_odpairs = pd.read_csv(DATA_PATH + DATA_FILE_OD_PAIRS, sep='|')

    df_d_types = pd.read_csv(DATA_PATH + DATA_FILE_D_TYPES, sep='|')

    avg_delays = pd.read_csv(DATA_PATH + DATA_FILE_DELAYS, sep='|')

    # Architecture parameters
    arch_params = {}
    arch_params['F'] = [None, 32, 16, 8]
    #arch_params['F'] = []
    arch_params['K'] = 3
    arch_params['M'] = [1024, 512, 64, 32, 2]
    arch_params['nonlin'] = nn.Tanh
    arch_params['nonlin_mlp'] = nn.ReLU
    arch_params['arch_info'] = ARCH_INFO
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

    for col in COLS:

        df = df_nodes if col == 'NODE' else df_odpairs

        entities = np.array(sorted(df[col].unique()))

        # Load the graph
        S = np.load(GRAPH_PATH + 'Adj_' + col.lower() + 's.npy', allow_pickle=True)
        S = S / np.abs(np.linalg.eigvals(S)).max()
        arch_params['S'] = S

        # Obtain data
        X, mlp_features = obtain_data(df, df_d_types, avg_delays, entities)
        if len(arch_params['F']) > 0:
            n_feats = X.shape[1]
            arch_params['F'][0] = n_feats

        df_work = df[df['YEAR'] == 2018].reset_index(drop=True)
        idx_work = len(df_work) // entities.shape[0]
        X_work = X[:idx_work,:,:]
        mlp_features_work = mlp_features[:idx_work,:]

        df_test = df[df['YEAR'] == 2019].reset_index(drop=True)
        X_test_unbal = X[idx_work:,:,:]
        mlp_features_test_unbal = mlp_features[idx_work:,:]

        arch_params['n_mlp_feat'] = mlp_features.shape[1]

        assert (len(df_test) // entities.shape[0]) == X_test_unbal.shape[0]

        results = {}
        
        procs = {}
        with Pool(processes=N_CPUS) as p:
            for ent in entities:
                print("Entity: {} - ".format(ent))
                procs[ent] = p.apply_async(test_entity, args=[df_work[df_work[col] == ent].reset_index(drop=True),
                                                             df_test[df_test[col] == ent].reset_index(drop=True),
                                                             mlp_features,
                                                             X_test_unbal,
                                                             mlp_features_test_unbal])
            for i, entity in enumerate(entities):
                results[entity] = procs[entity].get()
                print("Entity: {}, i: {}, Results: {}".format(entity, i, results[entity]['unbal']))

        with open(RESULTS_PATH + 'results_GNN_' + col.lower() + '.json', 'w') as f:
            json.dump(results, f)
