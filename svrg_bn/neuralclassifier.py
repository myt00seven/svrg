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

    input_layer  =  lasagne.layers.InputLayer(shape=(None, n_input), input_var=input_var)
    hidden_layer =  lasagne.layers.DenseLayer(input_layer, num_units=n_hidden,nonlinearity=lasagne.nonlinearities.rectify)        
    hidden_layer2 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(hidden_layer, num_units=n_hidden,nonlinearity=lasagne.nonlinearities.rectify))
    output_layer =  lasagne.layers.DenseLayer(hidden_layer2, num_units=n_output, nonlinearity=lasagne.nonlinearities.softmax)                    

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

        l1_reg = lasagne.regularization.regularize_layer_params(network, lasagne.regularization.l1)
        l2_reg = lasagne.regularization.regularize_layer_params(network, lasagne.regularization.l2)

        if (gradient == 'svrg'):
            loss = objective(prediction, self.target_var) + 0.01 * l2_reg
        elif (gradient == 'stream'):
            loss = objective(prediction, self.target_var) + 0.1 * l1_reg        
        elif (gradient == 'adagrad'):
            loss = objective(prediction, self.target_var) + 0.1 * l1_reg        

        # loss = objective(prediction, self.target_var) + lambd * l1_reg
        loss = loss.mean()
    
        params = lasagne.layers.get_all_params(network, trainable=True)

 #       svrg = False
        
    
        if (update == custom_svrg1):
            optimizer = SVRGOptimizer(update_params['m'], update_params['learning_rate'],update_params['adaptive'], update_params['adaptive_half_life_period'])
            # m is fixed as 50
            train_error, validation_error, acc_train, acc_val, acc_test, test_error, epoch_times = optimizer.minimize(loss, params,
                    X_train, Y_train, X_test, y_test, 
                    self.input_var, self.target_var, 
                    X_val, Y_val, 
                    n_epochs=n_epochs, batch_size=batch_size, output_layer=network)
            
        elif (update == custom_streaming_svrg1):
            optimizer = StreamingSVRGOptimizer(update_params['m'], update_params['learning_rate'], update_params['k_s_0'], update_params['k_s_ratio'],update_params['adaptive'], update_params['adaptive_half_life_period'])
            train_error, validation_error, acc_train, acc_val, acc_test, test_error, epoch_times = optimizer.minimize(loss, params,
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
                test_acc_fn = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var),dtype=theano.config.floatX)
                val_fn = theano.function([self.input_var, self.target_var], [test_loss, test_acc_fn])
            else:
                val_fn = None

            test_error = []
            acc_test = []
            # these two are not realized yet
            
            train_error, validation_error, acc_train, acc_val, acc_test, test_error, epoch_times = train(
                    X_train, Y_train, X_val, Y_val, X_test, y_test,
                    train_fn, val_fn,
                    n_epochs, batch_size=batch_size, **update_params#, toprint=it
            )

        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_loss_train.txt",train_error)
        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_loss_val.txt",map(itemgetter(0), validation_error))
        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_loss_gradient_number.txt",map(itemgetter(1),validation_error))
        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_loss_test.txt",test_error)

        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_acc_train.txt",acc_train)
        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_acc_val.txt",acc_val)
        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_acc_test.txt",acc_test)
        np.savetxt("data/""_mlpbn"+str(MLPBN)+"_"+ gradient +"_epoch_times.txt",epoch_times)
        

        return train_error, validation_error
