import sys, os, time
import numpy as np

import theano
import theano.tensor as T

import lasagne

import matplotlib.pyplot as plt
import seaborn
import time

from custom_updates import *

from load_dataset import *

import neuralclassifier

BATCH_SIZE = 100
NUM_EPOCHS = 500
NUM_HIDDEN_UNITS = 500
GRADIENT = "svrg"
MODEL = "MLPBN"

# as the training set of MNIST is 50000, set the batch size to 100 means it taks 500 batches to go through the entire training set

def main(model=MODEL,gradient = GRADIENT, n_epochs=NUM_EPOCHS, n_hidden = NUM_HIDDEN_UNITS):

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # n_epochs = 1000
    # n_hidden = 500

    objective = lasagne.objectives.categorical_crossentropy

    models = {}
    l_r = theano.shared(np.array(0.01, dtype="float32")) 
    ada_factor = theano.shared(np.array(1.0, dtype="float32")) 
    if gradient == "svrg" or gradient == "all":
        models.update({ 'svrg': (custom_svrg1, {'learning_rate': 0.01, 'm': 50, 'adaptive': True, 'adaptive_half_life_period':20, 'ada_factor':ada_factor}) })
    if gradient == "stream" or gradient == "all": # It is StreamingSVRG
        models.update({ 'streaming': (custom_streaming_svrg1, {'learning_rate': 0.1, 'm': 50, 'k_s_0': 1.0, 'k_s_ratio':1.03, 'adaptive': True, 'adaptive_half_life_period':20, 'ada_factor':ada_factor}) })
        #k_s is the ratio of how many batches are used in this iteration of StreamingSVRG
    if gradient == "adagrad" or gradient == "all":        
        models.update( { 'adagrad': (custom_adagrad, {'learning_rate': l_r, 'eps': 1.0e-8, 'adaptive': True, 'adaptive_half_life_period':20, 'ada_factor':ada_factor}) })



    # print(models.keys())

    # models = {
    #     # 'svrg_classif': (custom_svrg1, {'learning_rate': 0.01, 'm': 50})
    #     # 'adam_classif': (custom_adam, {'learning_rate': 0.01})
    #    # 'adam_classif_dropout': (lasagne.updates.adam, {'learning_rate': 0.01})
    # }

    for model in models.keys():
        update, update_params = models[model]

        np.random.seed(19921010)

        network = neuralclassifier.NeuralClassifier(n_input=X_train.shape[1], n_hidden=n_hidden, n_output=10)

        train_err, val_err = network.train(X_train, y_train, X_val, y_val, X_test, y_test,
                                           n_epochs=n_epochs, lambd=0.1,
                                           objective=objective, update=update, batch_size=BATCH_SIZE, gradient=model,  **update_params )

    #     np.savez('models/model_%s.npz' % model, *lasagne.layers.get_all_param_values(network.output_layer))
    #     np.savez('models/model_%s_val_error.npz' % model, val_err)

    # plt.title('Validation error/epoch')    
    # plt.legend()
    # plt.show()
        

if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['model'] = sys.argv[1]
    if len(sys.argv) > 2:
        kwargs['gradient'] = sys.argv[2]
    if len(sys.argv) > 3:
        kwargs['n_epochs'] = int(sys.argv[3])
    if len(sys.argv) > 4:
        kwargs['n_hidden'] = int(sys.argv[4])
    main(**kwargs)    
