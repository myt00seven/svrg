"""
Main program
"""

import sys, os, time
import numpy as np

import theano
import theano.tensor as T

import lasagne

import matplotlib.pyplot as plt
import seaborn

from custom_updates import *

from load_dataset import *

import neuralclassifier

BATCH_SIZE= 100

USE_STREAMING_SVRG = 0

def main():
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    n_epochs = 1000 
    n_hidden = 500
 
    objective = lasagne.objectives.categorical_crossentropy

    models = {
    # According to Nocedal's paper, m should be 2n~5n
        'svrg_classif': (custom_svrg1, {'learning_rate': 0.01, 'm': 2*BATCH_SIZE, 'streaming': USE_STREAMING_SVRG})
    # straming control if we use straming SVRG

    # Right here we have adam!

    #'adam_classif': (custom_adam, {'learning_rate': 0.01})
    #'adam_classif_dropout': (lasagne.updates.adam, {'learning_rate': 0.01})
    }

    for model in models.keys():
        update, update_params = models[model]

        network = neuralclassifier.NeuralClassifier(n_input=X_train.shape[1], n_hidden=n_hidden, n_output=10)

        train_err, val_err = network.train(X_train, y_train, X_val, y_val, X_test, y_test,
                                           n_epochs=n_epochs, lambd=0.0,
                                           objective=objective, update=update, batch_size=BATCH_SIZE, **update_params )

#       if type(val_err[0]) == tuple:
#           y, x = zip(*val_err)
#           plt.plot(x, y, label=model)
#       else:
#           plt.plot(val_err, label=model)
  
        # np.savez('models/model_%s.npz' % model, *lasagne.layers.get_all_param_values(network.output_layer))
        # np.savez('models/model_%s_val_error.npz' % model, val_err)

    # plt.title('Validation error/epoch')    
    # plt.legend()
    # plt.show()
        

if __name__ == '__main__':
    main()
