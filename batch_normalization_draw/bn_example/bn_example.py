from __future__ import print_function

import gzip
import itertools
import pickle
import os
import sys

PY2 = sys.version_info[0] == 2

if PY2:
    from urllib import urlretrieve
    pickle_load = lambda f, encoding: pickle.load(f)
else:
    from urllib.request import urlretrieve
    pickle_load = lambda f, encoding: pickle.load(f, encoding=encoding)


import numpy as np
import lasagne
import theano
import theano.tensor as T
#theano.config.compute_test_value = 'raise'

DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = 'mnist.pkl.gz'

NUM_EPOCHS = 500
BATCH_SIZE = 100
NUM_HIDDEN_UNITS = 500
LEARNING_RATE = 0.01
MOMENTUM = 0.9


def _load_data(url=DATA_URL, filename=DATA_FILENAME):
    if not os.path.exists(filename):
        print("Downloading MNIST")
        urlretrieve(url, filename)

    with gzip.open(filename, 'rb') as f:
        return pickle_load(f, encoding='latin-1')


def load_data():
    data = _load_data()
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]

    return dict(
        X_train=theano.shared(lasagne.utils.floatX(X_train)),
        y_train=T.cast(theano.shared(y_train), 'int32'),
        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid=T.cast(theano.shared(y_valid), 'int32'),
        X_test=theano.shared(lasagne.utils.floatX(X_test)),
        y_test=T.cast(theano.shared(y_test), 'int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1],
        output_dim=10,
        )


def build_model(input_dim, output_dim,
                batch_size=BATCH_SIZE, num_hidden_units=NUM_HIDDEN_UNITS):

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, input_dim),
        )
    l_hidden1 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        l_in,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
        ))
    #l_hidden1 = lasagne.layers.DropoutLayer(
    #    l_hidden1,
    #    p=0.5,
    #    )
    l_hidden2 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
        ))
    #l_hidden2 = lasagne.layers.DropoutLayer(
    #    l_hidden2,
    #    p=0.5,
    #    )
    l_out = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        l_hidden2,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
        ))
    return l_out


def create_iter_functions(dataset, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM):
    batch_index = T.iscalar('batch_index')
    batch_index.tag.test_value = np.int32(3)
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')

    X_batch.tag.test_value = np.random.rand(BATCH_SIZE, 784).astype(theano.config.floatX)
    y_batch.tag.test_value = (np.random.rand(BATCH_SIZE, 10) > 0.3).astype('int32')

    batch_slice = slice(batch_index * batch_size, (batch_index + 1) * batch_size)

    train   =    lasagne.layers.get_output(output_layer, deterministic = False, batch_norm_update_averages=True)
    prediction = lasagne.layers.get_output(output_layer, deterministic = True)

    loss_train = T.mean(lasagne.objectives.categorical_crossentropy(train, y_batch))
    loss_eval  = T.mean(lasagne.objectives.categorical_crossentropy(prediction, y_batch))
    
    # create updates
    def batchnormalizeupdates(tensors, params, avglen):
        updates = []
        mulfac = 1.0/avglen
        for tensor, param in zip(tensors, params):
            updates.append((param, (1.0-mulfac)*param + mulfac*tensor))
        return updates

    batchupd = batchnormalizeupdates(batchnormvalues, batchnormparams, 100)
    pred = T.argmax(output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch))

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)

    # add batchnormalize updates
    updates += batchupd

    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
            },
        )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
            },
        )

    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
            },
        )

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
        )


def train(iter_funcs, dataset, batch_size=BATCH_SIZE):
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size
    num_batches_test = dataset['num_examples_test'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
            }


def main(num_epochs=NUM_EPOCHS):
    dataset = load_data()
    output_layer = build_model(
        input_dim=dataset['input_dim'],
        output_dim=dataset['output_dim'],
        )
    iter_funcs = create_iter_functions(dataset, output_layer)

    print("Starting training...")
    for epoch in train(iter_funcs, dataset):
        print("Epoch %d of %d" % (epoch['number'], num_epochs))
        print("  training loss:\t\t%.6f" % epoch['train_loss'])
        print("  validation loss:\t\t%.6f" % epoch['valid_loss'])
        print("  validation accuracy:\t\t%.2f %%" %
              (epoch['valid_accuracy'] * 100))

        if epoch['number'] >= num_epochs:
            break

    return output_layer


if __name__ == '__main__':
    main()