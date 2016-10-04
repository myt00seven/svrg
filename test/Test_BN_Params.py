import numpy as np

import theano
import theano.tensor as T

import lasagne

from neuralnet import train

from custom_updates import *
from SVRGOptimizer import SVRGOptimizer
from StreamingSVRGOptimizer import StreamingSVRGOptimizer
from operator import itemgetter

n_input = 128
num_units = 200
n_hidden  = 200
n_output = 10

input_layer  = lasagne.layers.InputLayer(shape=(None, n_input))

normal_layer = lasagne.layers.DenseLayer(input_layer ,num_units=n_hidden,nonlinearity=lasagne.nonlinearities.rectify)

hidden_layer =  lasagne.layers.DenseLayer(input_layer, num_units=n_hidden,nonlinearity=lasagne.nonlinearities.rectify)        
hidden_layer2 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(hidden_layer, num_units=n_hidden,nonlinearity=lasagne.nonlinearities.rectify))
output_layer =  lasagne.layers.DenseLayer(hidden_layer2, num_units=n_output, nonlinearity=lasagne.nonlinearities.softmax)                    

lasagne.layers.get_all_params(hidden_layer)
lasagne.layers.get_all_params(hidden_layer2)
lasagne.layers.get_all_params(output_layer)
lasagne.layers.get_all_params(output_layer, trainable = True)

lasagne.layers.get_all_param_values(output_layer)
lasagne.layers.get_all_param_values(output_layer, trainable = True)