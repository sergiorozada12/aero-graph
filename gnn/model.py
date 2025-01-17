from torch import optim, no_grad, nn, Tensor
import torch
import copy
import time
import numpy as np
import sys

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# Optimizer constans
SGD = 1
ADAM = 0


class Model:
    # TODO: add support for more optimizers
    def __init__(self, arch,
                 learning_rate=0.1, decay_rate=0.99, loss_func=nn.MSELoss(),
                 epochs=50, batch_size=100, eval_freq=5, verbose=False,
                 max_non_dec=10, opt=ADAM):
        assert opt in [SGD, ADAM], 'Unknown optimizer type'
        self.arch = arch
        self.loss = loss_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.verbose = verbose
        self.max_non_dec = max_non_dec
        if opt == ADAM:
            self.optim = optim.Adam(self.arch.parameters(), lr=learning_rate)
        else:
            self.optim = optim.SGD(self.arch.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optim,
                                                          decay_rate)

    def count_params(self):
        return sum(p.numel() for p in self.arch.parameters() if p.requires_grad)

    def fit(self, train_X, train_Y, train_mlp_feat, val_X, val_Y, val_mlp_feat):
        n_samples = train_X.shape[0]
        n_steps = int(n_samples/self.batch_size)

        best_err = 1000000
        best_net = None
        cont = 0
        train_err = np.zeros(self.epochs)
        val_err = np.zeros(self.epochs)
        for i in range(1, self.epochs+1):
            t_start = time.time()
            for j in range(1, n_steps+1):
                # Randomly select batches
                idx = np.random.permutation(n_samples)[:self.batch_size]
                batch_X = train_X[idx, :, :]
                batch_mlp_feat = train_mlp_feat[idx,:]
                batch_Y = train_Y[idx]
                self.arch.zero_grad()

                # Training step
                predicted_Y = self.arch(batch_X, batch_mlp_feat)
                training_loss = self.loss(predicted_Y, batch_Y)
                # CHANGE!!!
                training_loss.backward()
                self.optim.step()

            self.scheduler.step()
            train_err[i-1] = training_loss.detach().numpy()
            t = time.time()-t_start

            # Predict eval error
            with no_grad():
                predicted_Y_eval = self.arch(val_X, val_mlp_feat)
                eval_loss = self.loss(predicted_Y_eval, val_Y)
                val_err[i-1] = eval_loss.detach().numpy()
            if eval_loss.data*1.005 < best_err:
                best_err = eval_loss.data
                best_net = copy.deepcopy(self.arch)
                cont = 0
            else:
                if cont >= self.max_non_dec:
                    break
                cont += 1

            if self.verbose and i % self.eval_freq == 0:
                print('Epoch {}/{}({:.4f}s)\tEval Loss: {:.8f}\tTrain: {:.8f}'
                      .format(i, self.epochs, t, eval_loss, training_loss))
        self.arch = best_net
        return i-cont, train_err, val_err

    def state_dict(self):
        return self.arch.state_dict()

    def load_state_dict(self, state_dict):
        self.arch.load_state_dict(state_dict)
        self.arch.eval()

    def print_model_state_sizes(self):
        for params in self.arch.state_dict():
            print(params, "\t", self.arch.state_dict()[params].requires_grad)

    def predict(self, X, mlp_feat=None):
        self.arch.eval()

        Y_hat = self.arch(X, mlp_feat)

        sm = nn.Softmax(dim=1)
        predictions = sm(Y_hat.detach()).argmax(dim=1)

        return predictions

    def test(self, test_X, test_Y, test_mlp_feat):
        self.arch.eval()
        n_samples = test_Y.shape[0]

        # Loss
        Y_hat = self.arch(test_X, test_mlp_feat)
        loss = self.loss(Y_hat, test_Y)

        sm = nn.Softmax(dim=1)
        predictions = sm(Y_hat.detach()).argmax(dim=1)
        accuracy = torch.eq(test_Y, predictions).sum().to(dtype=torch.float) / n_samples
        precision = precision_score(test_Y, predictions)
        recall = recall_score(test_Y, predictions)

        return loss.detach().item(), accuracy.detach().item(), precision, recall


class LinearModel:
    def __init__(self, N, loss_func=nn.MSELoss(), verbose=False):
        self.Beta = np.zeros((N, N))
        self.loss = loss_func
        self.verbose = verbose

    def count_params(self):
        return self.Beta.shape[0]**2

    def fit(self, train_X, train_Y, val_X=None, val_Y=None):
        X = train_X.view([train_X.shape[0], train_X.shape[2]]).detach().numpy()
        Y = train_Y.view([train_Y.shape[0], train_Y.shape[2]]).detach().numpy()
        self.Beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y)
        return 0, 0, 0

    def state_dict(self):
        return self.Beta

    def load_state_dict(self, state_dict):
        self.Beta = state_dict

    def print_model_state_sizes(self):
        print('Beta\t', self.Beta.size())

    def test(self, test_X, test_Y):
        shape_Y = [test_Y.shape[0], test_Y.shape[2]]
        shape_X = [test_X.shape[0], test_X.shape[2]]
        X = test_X.view(shape_X).detach().numpy()
        Y = test_Y.view(shape_Y).detach().numpy()
        Y_hat = X.dot(self.Beta)

        mse = self.loss(Tensor(Y_hat), test_Y.view(shape_Y))
        err = np.sum((Y_hat-Y)**2, axis=1)/np.linalg.norm(Y, axis=1)**2
        return np.mean(err), np.median(err), mse.detach().numpy()
