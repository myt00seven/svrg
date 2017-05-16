#!/usr/bin/env python
from __future__ import print_function


"""
Batch Normalization + SVRG on MNIST
Independent Study
May 24, 2016
Yintai Ma
"""

"""
# Use SGD AdaGrad instead
"""

"""
under folder of batch_normalization
Before merge; number 1
have options for "mlp", "mlpbn"; "sgd" and "custom_svrg2" and "sgd_adagrad"
"""


import sys
import os
import time

import matplotlib
import matplotlib.pyplot as plt
# parameters for Linux
plt.switch_backend('agg')


import pylab 

#THEANO_FLAGS='floatX=float32,device=gpu0,mode=FAST_RUN'
import numpy as np 
import theano
import theano.tensor as T
import lasagne

import pickle
import gzip

import my_bn_layer # Define DBN, e.g., DBN1
import my_bn_layer2 # Define DBN2, e.g. ,1/m^2 MA
import my_bn_layer_const # Define for const alpha (actually we can just use the my_bn_layer)
# import my_bn_layer_5_10_m # Define DBN2, e.g. ,5/10+m MA

from collections import OrderedDict

# May 18, 2016, Yintai Ma
# standard setting , epoch = 500, batch size = 100

OUTPUT_FIGURE_PATH = 'data_large/'
OUTPUT_DATA_PATH = 'data_large/'
NUM_EPOCHS = 20
BATCH_SIZE = 100
NUM_HIDDEN_UNITS = 500
LEARNING_RATE = 0.01
MOMENTUM = 0.9
FREQUENCY = 0.1

MODEL = 'mlpbn'
GRADIENT = 'sgd_adagrad'
BNALG = 'original'

bnalg_const_dict = {
"const1":     1.0, 
"const075":   0.75, 
"const05":    0.5, 
"const025":   0.25, 
"const01":    0.1, 
"const001":   0.01, 
"const0001":  0.001, 
"const0":     0.0, }    


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def custom_svrg2(loss, params, m, learning_rate=0.01, objective=None, data=None, target=None, getpred=None):

    theano.pp(loss)
    
    grads = theano.grad(loss, params)
    n = data.shape[0]

    updates = OrderedDict()
    rng = T.shared_randomstreams.RandomStreams(seed=149)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)        
        mu = grad / n

        def oneStep(w):
            t = rng.choice(size=(1,), a=n)

            loss_part_tilde = objective(getpred(data[t], param), target[t])
            loss_part_tilde = loss_part_tilde.mean()
            g_tilde = theano.grad(loss_part_tilde, param)
        
            loss_part = objective(getpred(data[t], w), target[t])
            loss_part = loss_part.mean()
            g = theano.grad(loss_part, w)

            w = w - learning_rate * (g - g_tilde + mu)
            return w

        w_tilde, scan_updates = theano.scan(fn=oneStep, outputs_info=param, n_steps=m)

        updates.update(scan_updates)
        updates[param] = w_tilde[-1]

    return updates

def mysgd(loss_or_grads, params, learning_rate):
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    return updates

def mysgd_adagrad(loss_or_grads, params, learning_rate=0.01, eps=1.0e-8):
    
    if not isinstance(loss_or_grads, list):
        grads = theano.grad(loss_or_grads, params)
    else:
        grads = loss_or_grads

    updates = OrderedDict()    

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        acc = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
        
        acc_new = acc + grad ** 2

        updates[acc] = acc_new
        updates[param] = param - learning_rate * grad / T.sqrt(acc_new + eps)

    return updates

def mysvrg(loss_or_grads, params, learning_rate,avg_gradient):
    #Not Working right now
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * (grad- grad_it + avg_gradient[param])

    return updates



def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    CIFAR_PATH = "../../data/cifar-10-batches-py/"

    def load_CIFAR_batch(filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            X = datadict['data']
            Y = datadict['labels']
            # print X.shape
            #X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
            X = X.reshape(10000, 3, 32, 32).transpose(0,1,2,3).astype("float32")
            Y = np.array(Y).astype("int32")
            return X, Y

    # We can now download and read the training and test set images and labels.
    # X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    # y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    # X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    # y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # Read NI dataset
    # DATA_PATH = "/Users/myt007/git/svrg/ni/"

    X_train1,y_train1 = load_CIFAR_batch(CIFAR_PATH + "data_batch_1")
    X_train2,y_train2 = load_CIFAR_batch(CIFAR_PATH + "data_batch_2")
    X_train3,y_train3 = load_CIFAR_batch(CIFAR_PATH + "data_batch_3")
    X_train4,y_train4 = load_CIFAR_batch(CIFAR_PATH + "data_batch_4")
    X_train5,y_train5 = load_CIFAR_batch(CIFAR_PATH + "data_batch_5")
    X_test,y_test = load_CIFAR_batch(CIFAR_PATH + "test_batch")

    X_train = np.concatenate((X_train1,X_train2,X_train3,X_train4,X_train5))
    y_train = np.concatenate((y_train1,y_train2,y_train3,y_train4,y_train5))

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    X_train = X_train.astype("float32")
    X_val   = X_val.astype("float32")
    X_test  = X_test.astype("float32")
    y_train = y_train.astype("int32")
    y_val   = y_val.astype("int32")
    y_test  = y_test.astype("int32")

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    # print("X_train1")
    # print(X_train1.shape)
    # print(X_train1.dtype)
    # print("\n")
    # print("y_train1")
    # print(y_train1.shape)
    # print("X_train")
    # print(X_train.shape)
    # print(X_train.dtype)
    # print("\n")
    # print("y_train")
    # print(y_train.shape)
    # # print("X_val")
    # # print(X_val.shape)
    # # print("y_val")
    # # print(y_val.shape)
    # print("X_test")
    # print(X_test.shape)
    # print("y_test")
    # print(y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(input_var=None, num_hidden_units=NUM_HIDDEN_UNITS):
    l_in = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                     input_var=input_var)
    l_hid = lasagne.layers.DenseLayer(
            l_in, num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_hid = lasagne.layers.DenseLayer(
            l_hid, num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_out = lasagne.layers.DenseLayer(
            l_hid, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def build_mlpbn(input_var=None, num_hidden_units=NUM_HIDDEN_UNITS,bnalg = BNALG):

    

    l_in = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                     input_var=input_var)

    if bnalg == 'original':
        l_hidden = lasagne.layers.batch_norm (
            lasagne.layers.DenseLayer(
            l_in,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            )
        )
        l_hidden = lasagne.layers.batch_norm (
            lasagne.layers.DenseLayer(
            l_hidden,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            )
        )
    elif bnalg == 'dbn':
        l_hidden = my_bn_layer.my_batch_norm (
            lasagne.layers.DenseLayer(
            l_in,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            )
        )
        l_hidden = my_bn_layer.my_batch_norm (
            lasagne.layers.DenseLayer(
            l_hidden,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            )
        )
    elif bnalg == 'dbn2':
        l_hidden = my_bn_layer2.my_batch_norm (
        lasagne.layers.DenseLayer(
        l_in,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
            )
        )
        l_hidden = my_bn_layer2.my_batch_norm (
            lasagne.layers.DenseLayer(
            l_hidden,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            )
        )
    elif 'const' in bnalg:
        # print(bnalg)
        # print(bnalg_const_dict)

        if bnalg not in bnalg_const_dict:
            print("Incorrect bnalg method. Can't find in predefined dictionary.")
        else:

            the_alpha = bnalg_const_dict[bnalg]

            l_hidden = my_bn_layer_const.my_batch_norm (
            lasagne.layers.DenseLayer(
            l_in,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
                ), alpha = the_alpha            
            )
            l_hidden = my_bn_layer_const.my_batch_norm (
                lasagne.layers.DenseLayer(
                l_hidden,
                num_units=num_hidden_units,
                nonlinearity=lasagne.nonlinearities.rectify,
                ), alpha = the_alpha
            )

    l_out = lasagne.layers.DenseLayer(
        l_hidden,
        num_units=NUM_HIDDEN_UNITS,
        nonlinearity=lasagne.nonlinearities.softmax,
    )        
    
    return l_out


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False ):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(model=MODEL,gradient = GRADIENT, num_epochs=NUM_EPOCHS, num_hidden_units = NUM_HIDDEN_UNITS, bnalg = BNALG):
    rng = np.random.RandomState(42)
    lasagne.random.set_rng(rng)

    # Load the dataset
    NUM_EPOCHS = num_epochs
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(input_var, num_hidden_units)
    elif model == 'mlpbn':
        network = build_mlpbn(input_var, num_hidden_units,bnalg)
    else:
        print("Unrecognized model type %r." % model)
        return

    prediction = lasagne.layers.get_output(network, deterministic= False, batch_norm_update_averages = True)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(prediction, target_var))
    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),dtype=theano.config.floatX)
    
    params = lasagne.layers.get_all_params(network, trainable=True)

    if gradient == 'sgd':
        updates = mysgd(loss, params, LEARNING_RATE)
    elif gradient == 'sgd_adagrad':
        updates = mysgd_adagrad(loss, params, LEARNING_RATE)        
    elif gradient == 'svrg':
        updates = custom_svrg2(loss,params, m=100, learning_rate = LEARNING_RATE, objective=lasagne.objectives.categorical_crossentropy , data=input_var, target = target_var, getpred= getpred)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    train_acc = theano.function([input_var, target_var], acc)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:

    loss_train   = []
    loss_val =  []
    acc_val = []
    acc_train = []
    acc_test = []
    loss_test = []
    times = []

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        tmp_acc = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            tmp_acc += train_acc(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, BATCH_SIZE, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # and a full pass over the test data, bingo!
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1

        # Then we print the results for this epoch:
        times.append(time.time() - start_time)
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        loss_train.append(train_err / train_batches)
        acc_train.append(tmp_acc / train_batches)        
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        loss_val.append(val_err / val_batches)
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        acc_val.append(val_acc / val_batches)
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        acc_test.append(test_acc / test_batches)
        loss_test.append(test_err / test_batches)
        print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
        # print("  test loss:\t\t{:.2f}".format(test_err / val_batches))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
    print("  average time per epoch :\t\t{:.3f} s".format(np.mean(times)))


    print("result/"+model+"_"+gradient+"_"+bnalg+".txt")
    file_handle=open("result/"+model+"_"+gradient+"_"+bnalg+".txt", 'w+')
    file_handle.write("Final results:\n")
    file_handle.write("  test loss:\t\t\t{:.6f}\n".format(test_err / test_batches))
    file_handle.write("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
    
    count = (np.arange(NUM_EPOCHS)+1) #*X_train.shape[0]

    #PLOT 
    matplotlib.rcParams.update({'font.size': 16})
    plt.figure(1)
    plt.plot(count, loss_train, 'bs-',label="Training Set")
    plt.title(model+'-'+gradient+'-Loss of Training/Validation Set')
    plt.plot(count, loss_val, 'ro--',label="Validation Set")
    plt.xlabel('# Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    pylab.savefig(OUTPUT_FIGURE_PATH+'fig_LossTrain-'+model+'-'+gradient+'-'+str(NUM_EPOCHS)+'.png',
       bbox_inches='tight')
    
    plt.figure(2)
    plt.plot(count, acc_train, 'bs-',label="Training Set")
    plt.title(model+'-'+gradient+'-Predict Accuracy of Training/Validation Set')
    plt.plot(count, acc_val, 'ro--',label="Validation Set")
    plt.plot(count, acc_test, 'g^:',label="Test Set")
    plt.xlabel('# Epochs')
    plt.ylabel('Predict Accuracy')
    plt.legend(bbox_to_anchor=(1,0.25))
    # plt.show()
    pylab.savefig(OUTPUT_FIGURE_PATH+'fig_Pred-'+model+'-'+gradient+'-'+str(NUM_EPOCHS)+'.png',
       bbox_inches='tight')

    print ("Finish plotting...")

    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(NUM_EPOCHS)+"_"+bnalg+"_"+"loss_train.txt",loss_train)
    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(NUM_EPOCHS)+"_"+bnalg+"_"+"loss_val.txt",loss_val)
    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(NUM_EPOCHS)+"_"+bnalg+"_"+"acc_train.txt",acc_train)
    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(NUM_EPOCHS)+"_"+bnalg+"_"+"acc_val.txt",acc_val)
    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(NUM_EPOCHS)+"_"+bnalg+"_"+"acc_test.txt",acc_test)
    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(NUM_EPOCHS)+"_"+bnalg+"_"+"loss_test.txt",loss_test)
    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(NUM_EPOCHS)+"_"+bnalg+"_"+"epoch_times.txt",times)

    print ("Data saved...")


    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv) or ('help' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL] [GRADIENT] [NUM_EPOCHS]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'mlpbn: for an MLP with batch Normalization")
        print("GRADIENT: 'sgd', 'svrg'")
        print("NUM_EPOCHS: ")
        print("NUM_HIDDEN_UNITS: ")
        print("BNALG: ")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['gradient'] = sys.argv[2]
        if len(sys.argv) > 3:
            kwargs['num_epochs'] = int(sys.argv[3])
        if len(sys.argv) > 4:
            kwargs['num_hidden_units'] = int(sys.argv[4])
        if len(sys.argv) > 5:
            kwargs['bnalg'] = sys.argv[5]
        main(**kwargs)

