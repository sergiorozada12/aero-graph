import torch.nn as nn
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split

from gnn.model import Model, ADAM
from gnn.arch import BasicArch


DATA_PATH = 'E:/TFG_VictorTenorio/Aero_TFG/features/'
FEATSEL_PATH = 'E:/TFG_VictorTenorio/Aero_TFG/modelIn/'
GRAPH_PATH = 'E:/TFG_VictorTenorio/Aero_TFG/graph/'
RESULTS_PATH = 'E:/TFG_VictorTenorio/Aero_TFG/results/'
FEATURES_FILE = "features_lr.json"

VERB = False
ARCH_INFO = True


def eval_arch(X_train, y_train, X_val, y_val, X_test, y_test):
    archit = BasicArch(**arch_params)
    model_params['arch'] = archit

    print("Started Training")

    model = Model(**model_params)
    epochs, train_err, val_err = model.fit(X_train, y_train, X_val, y_val)

    print("Finished Training Model")
    print("Epochs: {} - Train Error: {} - Validation Error: {}".format(
        epochs, train_err, val_err
    ))

    print("Started Testing")

    mean_norm_error, median_norm_error, node_mse = model.test(X_test, y_test)

    print("----- END -------")
    print("Mean Norm Error: {} - Median Norm Error: {} - Node MSE: {}".format(
        mean_norm_error, median_norm_error, node_mse
    ))

    results = {
        'epochs': epochs,
        'train_err': train_err,
        'val_err': val_err,
        'node_mse': node_mse,
        'mean_norm_error': mean_norm_error,
        'median_norm_error': median_norm_error
    }

    return results


# Load the signal
df = pd.read_csv(DATA_PATH + 'incoming_delays_nodes.csv', sep='|')
n_feats = len(df.columns) - 2       # y and NODE

avg_delays = pd.read_csv(DATA_PATH + 'avg_delays.csv', sep='|')

# Load the graph
S = np.load(GRAPH_PATH + 'Adj_nodes.npy', allow_pickle=True)

# Architecture parameters

arch_params = {}
arch_params['S'] = S
arch_params['F'] = [n_feats, 32, 16, 8]
arch_params['K'] = 3
arch_params['M'] = [32, 2]
arch_params['nonlin'] = nn.Tanh
arch_params['arch_info'] = ARCH_INFO

model_param = {}

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

COL = 'NODE'

with open(FEATSEL_PATH + FEATURES_FILE, 'r') as f:
    features = json.load(f)

entities = np.array(sorted(df[COL].unique()))

results = {}
for ent in entities:
    print("Entity: {} - ".format(ent), end="")
    df_ent = df[df[COL] == ent]

    df_ent = pd.concat([df_ent, avg_delays], axis=1)
    X = df_ent.loc[:, features[ent]["cols"]].values

    y = df_ent['y_clas'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.12, random_state=42)

    results[ent] = eval_arch(X_train, y_train, X_val, y_val, X_test, y_test)
    print("DONE - Median Error: {}".format(results[ent]['mean_norm_error']))