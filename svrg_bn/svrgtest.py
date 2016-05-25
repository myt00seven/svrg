import sys, os, time
import numpy as np

import theano
import theano.tensor as T

import lasagne

import matplotlib.pyplot as plt
import seaborn

from custom_updates import *

from load_dataset import *

import autoencoder
import deep

NUM_EPOCHS = 20
BATCH_SIZE = 100
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9
FREQUENCY = 0.1

def main():
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    X_train = X_train[:]

    n_epochs = 20
    n_hidden = 100

    objective = lasagne.objectives.binary_crossentropy
    #objective = lasagne.objectives.squared_error

    models = {
    #    'sdg_test_long_nomomentum': (custom_momentum, {'learning_rate': 10.0, 'momentum': 0.0}),
        'svrg_testing_nonadaptive_withprob': (custom_svrg1, {'learning_rate': 128.0, 'm': 500, 'adaptive': False})
        # 'svrg_testing_nonadaptive_withprob': (custom_svrg1, {'learning_rate': 128.0, 'm': 500, 'adaptive': False})
    #    'momentum_1.0_0.9_300': (custom_momentum, {'learning_rate': 1.0, 'momentum': 0.9}),
    #    'adam_test_faster100epochs': (custom_adam, {'learning_rate': 0.01}),
    }

    for model in models.keys():
        update, update_params = models[model]

        network = autoencoder.DenoisingAutoEncoder(n_input=X_train.shape[1], n_hidden=NUM_HIDDEN_UNITS)
        # network = build_mlp(input_var, X_train.shape[1])

        train_err, val_err = network.train(X_train, X_val, n_epochs=n_epochs, batch_size=100, lambd=0.0,
                                           objective=objective, update=update, **update_params)

        if type(val_err[0]) == tuple:
            x, y = zip(*val_err)
            plt.plot(y, x, label=model)
        else:
            plt.plot(val_err, label=model)
    
        np.savez('models/model_%s.npz' % model, *lasagne.layers.get_all_param_values(network.output_layer))
        np.savez('models/model_%s_val_error.npz' % model, val_err)

    plt.title('Validation error/epoch')    
    plt.legend()
    plt.show()
        

if __name__ == '__main__':
    main()
