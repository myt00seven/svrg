#!/usr/bin/env python

"""
Batch Normalization + SVRG on MNIST
Independent Study
May 24, 2016
Yintai Ma
"""

"""
under folder of batch_normalization
Before merge; number 1
have options for "mlp", "mlpbn"; "sgd" and "custom_svrg2"
"""

from __future__ import print_function

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
MODEL = 'mlp'
GRADIENT = 'sgd'

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

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(input_var=None, num_hidden_units=NUM_HIDDEN_UNITS):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
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

def build_mlpbn(input_var=None, num_hidden_units=NUM_HIDDEN_UNITS):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
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
    l_out = lasagne.layers.batch_norm (
        lasagne.layers.DenseLayer(
        l_hidden,
        num_units=NUM_HIDDEN_UNITS,
        nonlinearity=lasagne.nonlinearities.softmax,
        )
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

def main(model=MODEL,gradient = GRADIENT, num_epochs=NUM_EPOCHS, num_hidden_units = NUM_HIDDEN_UNITS):
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
        network = build_mlpbn(input_var, num_hidden_units)
    else:
        print("Unrecognized model type %r." % model)
        return

    prediction = lasagne.layers.get_output(network, deterministic= False, batch_norm_update_averages = True)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(prediction, target_var))
    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),dtype=theano.config.floatX)
    
    params = lasagne.layers.get_all_params(network, trainable=True)

    if gradient == 'sgd':
        updates = mysgd(loss, params, LEARNING_RATE)
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

    for epoch in range(NUM_EPOCHS):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        tmp_acc = 0
        train_batches = 0
        start_time = time.time()
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
        acc_test.append(test_acc / val_batches)
        loss_test.append(test_err / val_batches)
        print("  test accuracy:\t\t{:.2f} %".format(test_acc / val_batches * 100))
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
    print("  average time per epoch :\t\t{:.3f} %".format(np.mean(times)))


    file_handle=open("result/"+model+"_"+gradient+".txt", 'w+')
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

    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(NUM_EPOCHS)+"_"+"loss_train.txt",loss_train)
    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(NUM_EPOCHS)+"_"+"loss_val.txt",loss_val)

    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(NUM_EPOCHS)+"_"+"acc_train.txt",acc_train)
    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(NUM_EPOCHS)+"_"+"acc_val.txt",acc_val)

    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(NUM_EPOCHS)+"_"+"acc_test.txt",acc_test)
    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(NUM_EPOCHS)+"_"+"loss_test.txt",loss_test)

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
        main(**kwargs)

