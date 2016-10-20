import sys, os, time
import numpy as np

import theano
import theano.tensor as T

import lasagne

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    #assert len(inputs) == len(targets)
    assert inputs.shape[0] == len(targets)
    if shuffle:
#        indices = np.arange(len(inputs))
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
#    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        # Output the entire training set, which output size of batchsize each time
        # Need to call iterate_minibatches from outside for n times to output the entire training set
        # n = training_set_size / batch_size
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def random_minibatches(inputs, targets, batchsize, k_s):
    # technically, if batchsize== certain dimension of X_train (inputs) , then no need to shuffle
    # for simplicity, here we just always shuffle

    assert inputs.shape[0] == len(targets)        
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, batchsize*k_s - batchsize + 1, batchsize):                
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]

def train(X_train, Y_train, X_val, Y_val, X_test, y_test, train_fn, val_fn, n_epochs, batch_size=500, verbose=True, toprint=None, **update_params):
    #train_fn includes the AdaGrad update functino

    train_error = []
    validation_error = []
    test_error =[]

    acc_train = []
    acc_val = []
    acc_test = []

    times = []
    epoch_times = []

    gradient_times = 0

    start_time = time.time()

    if verbose:
        print("Starting training...")
    for epoch in range(n_epochs):

        train_err = 0
        train_acc = 0
        train_batches = 0

        t = time.time()

        if epoch>0 and divmod(epoch,update_params['adaptive_half_life_period'])==0:
            # lr = update_params['learning_rate'].get_value()
            # lr = lr*update_params['ada_factor']
            update_params['learning_rate'].set_value(update_params['learning_rate'].get_value() * update_params['ada_factor'])

        # print ('...testing')
        # print '...testing'
        # print '...testing'

        print 'Learning Rate: ',update_params['learning_rate'].get_value()
        # print 'Learning Rate: ',update_params['learning_rate'].set_value(1)

        for batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
#            train_err += train_fn(np.array(inputs.todense(), dtype=np.float32), np.array(targets, dtype=np.int32))
            current_loss, current_acc = val_fn(inputs, targets)
            train_acc += current_acc    
            
            gradient_times += 1
            train_batches += 1

            if toprint is not None:
                print toprint.get_value()

        train_error.append(train_err / train_batches)
        acc_train.append(train_acc / train_batches)        

        if X_val is not None:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, Y_val, batch_size, shuffle=False):
                inputs, targets = batch
                # err = val_fn(inputs, targets)
                # err = val_fn(np.array(inputs.todense(), dtype=np.float32), np.array(targets, dtype=np.int32))

                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1
       
            validation_error.append((val_err / val_batches, gradient_times))
            acc_val.append(val_acc / val_batches)

        test_err = 0
        test_acc = 0
        test_batches = 0
        for i, batch in enumerate(iterate_minibatches(X_test, y_test, batch_size, shuffle=False)):
            inputs, targets = batch
            current_err, current_acc = val_fn(inputs, targets)
            test_err += current_err
            test_acc += current_acc
            test_batches += 1
        test_error.append(test_err / test_batches)
        acc_test.append(test_acc / test_batches)

        epoch_times.append(time.time()-start_time)        

        if verbose:
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, n_epochs, time.time() - t))
            # print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            if X_val is not None:
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
                print("  train accuracy:\t\t{:.6f}".format(train_acc / train_batches))
                print("  validation accuracy:\t\t{:.6f}".format(val_acc / val_batches))
                print("  test accuracy:\t\t{:.6f}\n".format(test_acc / test_batches))

        # train_error.append(train_err / train_batches)
        # validation_error.append((val_err / val_batches, self.counted_gradient.get_value()))

        # acc_train.append(train_acc / train_batches)
        

    return train_error, validation_error, acc_train, acc_val, acc_test, test_error, epoch_times
