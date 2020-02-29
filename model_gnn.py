import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

from gnn.model import Model, ADAM
from gnn.arch import BasicArch

DATA_PATH = 'C:/Users/victo/Aero_TFG/modelIn/'
RESULTS_PAT = 'C:/Users/victo/Aero_TFG/results/'

VERB = False
ARCH_INFO = True

# Load the signal
data = pd.read_csv(DATA_PATH + 'signal.csv', sep='|')

X = data.drop(columns='y').values
n_feats = X.shape[1]
y = data['y'].values

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# Load the graph
S = np.load(DATA_PATH + 'graph.csv')

# Architecture parameters

arch_params = {}
arch_params['S'] = S
arch_params['F'] = [n_feats, 32, 16, 8]
arch_params['K'] = 3
arch_params['M'] = [32, 1]
arch_params['nonlin'] = nn.Tanh
arch_params['arch_info'] = ARCH_INFO

archit = BasicArch(**arch_params)

model_param = {}

# Model parameters
model_params = {}
model_params['opt'] = ADAM
model_params['learning_rate'] = 0.001
model_params['decay_rate'] = 0.99
model_params['loss_func'] = nn.MSELoss()
model_params['epochs'] = 200
model_params['batch_size'] = 50
model_params['eval_freq'] = 4
model_params['max_non_dec'] = 10
model_params['verbose'] = VERB

print("Started Training")

model = Model(**model_param)
epochs, train_err, val_err = model.train(X_train, y_train, X_val, y_val)

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
