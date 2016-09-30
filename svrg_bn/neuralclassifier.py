import numpy as np

import theano
import theano.tensor as T

import lasagne

from neuralnet import train

from custom_updates import *
from SVRGOptimizer import SVRGOptimizer
from StreamingSVRGOptimizer import StreamingSVRGOptimizer
from operator import itemgetter

MLPBN= True

def classifier_network(input_var, n_input, n_hidden, n_output):

    input_layer  = lasagne.layers.InputLayer(shape=(None, n_input), input_var=input_var)
    hidden_layer = lasagne.layers.batch_norm(
            lasagne.layers.DenseLayer(
            input_layer, #            lasagne.layers.dropout(input_layer, p=0.5),
            num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
                    )
    hidden_layer = lasagne.layers.batch_norm(
            lasagne.layers.DenseLayer(
            hidden_layer, #            lasagne.layers.dropout(input_layer, p=0.5),
            num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)
                    )
    output_layer = lasagne.layers.batch_norm(
            lasagne.layers.DenseLayer(hidden_layer, num_units=n_output, nonlinearity=lasagne.nonlinearities.softmax)
                    )

    return input_layer, hidden_layer, output_layer

class NeuralClassifier:
    def __init__(self, n_input, n_hidden, n_output, input_var=None):
        self.n_input  = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.input_var = input_var or T.matrix('inputs')
        self.target_var = T.ivector('targets')
   
        self.input_layer, self.hidden_layer, self.output_layer = classifier_network(self.input_var, n_input, n_hidden, n_output)
    
    def train(self, X_train, Y_train, X_val=None, Y_val=None, X_test=None, y_test=None,
            objective=lasagne.objectives.binary_crossentropy, 
            update=lasagne.updates.adam, 
            n_epochs=100, batch_size=100, gradient="SVRG" , lambd=0.0,
            **update_params):

        network = self.output_layer

        prediction = lasagne.layers.get_output(network)

        l2_reg = lasagne.regularization.regularize_layer_params(network, lasagne.regularization.l2)
        loss = objective(prediction, self.target_var) + lambd * l2_reg
        loss = loss.mean()
    
        params = lasagne.layers.get_all_params(network, trainable=True)

 #       svrg = False
        
    
        if (update == custom_svrg1):
            optimizer = SVRGOptimizer(update_params['m'], update_params['learning_rate'])
            train_error, validation_error, acc_train, acc_val, acc_test, test_error = optimizer.minimize(loss, params,
                    X_train, Y_train, X_test, y_test, 
                    self.input_var, self.target_var, 
                    X_val, Y_val, 
                    n_epochs=n_epochs, batch_size=batch_size, output_layer=network)
            
        elif (update == custom_streaming_svrg1):
            optimizer = StreamingSVRGOptimizer(update_params['m'], update_params['learning_rate'], update_params['k_s'])
            train_error, validation_error, acc_train, acc_val, acc_test, test_error = optimizer.minimize(loss, params,
                    X_train, Y_train, X_test, y_test, 
                    self.input_var, self.target_var, 
                    X_val, Y_val, 
                    n_epochs=n_epochs, batch_size=batch_size, output_layer=network)

        else: # The AdaGrad version of SGD
            updates = update(loss, params, **update_params)

            train_fn = theano.function([self.input_var, self.target_var], loss, updates=updates)
        
            if X_val is not None:
                test_prediction = lasagne.layers.get_output(network, deterministic=True)
                test_loss = objective(test_prediction, self.target_var)
                test_loss = test_loss.mean()
                val_fn = theano.function([self.input_var, self.target_var], test_loss)
            else:
                val_fn = None
            
            train_error, validation_error = train(
                    X_train, Y_train, X_val, Y_val,
                    train_fn, val_fn,
                    n_epochs, batch_size=batch_size#, toprint=it
            )

        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_loss_train.txt",train_error)
        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_loss_val.txt",map(itemgetter(0), validation_error))
        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_loss_gradient_number.txt",map(itemgetter(1),validation_error))
        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_acc_train.txt",acc_train)
        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_acc_val.txt",acc_val)
        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_acc_test.txt",acc_test)
        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_loss_test.txt",test_error)

        return train_error, validation_error
