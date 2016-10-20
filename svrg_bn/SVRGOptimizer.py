import numpy as np

import theano
import theano.tensor as T
import lasagne

from theano.ifelse import ifelse

from neuralnet import iterate_minibatches

from collections import OrderedDict
import time

EXTRA_INFO = True
DEFAULT_ADAPTIVE = False
STREAMING_SVRG = False
DEMINISHING = True

current_factor = theano.shared(np.array(1, dtype="float32")) 
ada_factor = theano.shared(np.array(0.8, dtype="float32")) 

DEBUG_PARA = False # Debug of parameters insides the layers

class SVRGOptimizer:
    def __init__(self, m, learning_rate, adaptive=DEFAULT_ADAPTIVE, non_uniform_prob=True):
        self.m = m
        self.learning_rate = learning_rate
        self.adaptive = adaptive
        self.non_uniform_prob = non_uniform_prob

        self.counted_gradient = theano.shared(0)

    def minimize(self, loss, params, X_train, Y_train, X_test, y_test, input_var, target_var, X_val, Y_val, n_epochs=1000, batch_size=100, output_layer=None ):
        self.input_var = input_var
        self.target_var = target_var

        num_batches = X_train.shape[0] / batch_size
        n = num_batches

        if EXTRA_INFO:
            flog = open("data/log.txt",'w')        
            flog.write("Learning Rate:{:.2f}\n".format(self.learning_rate))
            flog.write("Adaptive:{:.2f}\n".format(self.adaptive))
            flog.write("Non Uniform Prob of mini batch:{:.2f}\n\n".format(self.non_uniform_prob))
        if DEBUG_PARA:
            fpara_train = open("data/log_para_train.txt",'w')        
            fpara_bn    = open("data/log_para_BN.txt",'w')        

        self.L = theano.shared(np.cast['float32'](1. / self.learning_rate))
#        self.Ls = [theano.shared(np.cast['float32'](1. / self.learning_rate)) for _  in range(num_batches)]
        self.Ls = [1. / self.learning_rate for _  in range(num_batches)]

        w_updates, mu_updates = self.make_updates(loss, params)

        # Update function for batch normalization layer
        train_bn = lasagne.layers.get_output(output_layer, deterministic=False, batch_norm_update_averages = True)
        loss_bn  = T.mean(lasagne.objectives.categorical_crossentropy(train_bn, target_var))
        update_bn_fn = theano.function([self.input_var, self.target_var], loss_bn)

        train_mu = theano.function([self.input_var, self.target_var], loss, updates=mu_updates)
        train_w = theano.function([self.input_var, self.target_var], loss, updates=w_updates)


        prediction = lasagne.layers.get_output(output_layer, deterministic=True)
        test_acc_fn = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),dtype=theano.config.floatX)

        val_fn = theano.function([self.input_var, self.target_var], [loss, test_acc_fn])


        train_error = []
        validation_error = []
        acc_train = []
        acc_val = []
        test_error = []
        acc_test = []
        times = []
        epoch_times = []

        print "NUMBATCHES: ", n

        j = 0

        L_fn = self.make_L_fn(loss, params)
        start_time = time.time()

        print("Starting training...")
        for epoch in range(n_epochs):

            if DEMINISHING and divmod(epoch,2)[1]==0 and epoch>0:
                    current_factor.set_value(np.float32(current_factor.get_value() / ada_factor.get_value() ) ) 

            if EXTRA_INFO:
                flog.write("Epoch:{:.2f}\n".format(epoch))
            if DEBUG_PARA:
                fpara_train.write("\nEpoch:{:.2f}\n".format(epoch))
                train_para_list = lasagne.layers.get_all_params(output_layer, trainable=True)
                for item in train_para_list:
                    fpara_train.write("%s\n" % item)
                train_para_list = lasagne.layers.get_all_param_values(output_layer, trainable=True)
                for item in train_para_list:
                    fpara_train.write("%s\n" % item)

                fpara_bn.write("\nEpoch:{:.2f}\n".format(epoch))
                bn_para_list = lasagne.layers.get_all_params(output_layer, trainable=False)
                for item in bn_para_list:
                    fpara_bn.write("%s\n" % item)
                bn_para_list = lasagne.layers.get_all_param_values(output_layer, trainable=False)
                for item in bn_para_list:
                    fpara_bn.write("%s\n" % item)

            t = time.time()

            train_err = 0
            train_acc = 0
            train_batches = 0

            if self.non_uniform_prob:
                batches = self.iterate_minibatches(X_train, Y_train, batch_size)
            else:
                batches = iterate_minibatches(X_train, Y_train, batch_size, shuffle=True)
            for batch in batches:
                if j % self.m == 0:
                    # When updating w_t for every m times, compute a \mu_tilde as an average directin over all training samples
                    for mu in self.mu:
                        mu.set_value(0 * mu.get_value())

                    for mu_batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=False):
                        inputs, targets = mu_batch
                        train_mu(inputs, targets)
                    
                    for mu in self.mu:
                        mu.set_value(mu.get_value() / n)

                j += 1               
                inputs, targets = batch
                #print "learning_rate: ", 1. / self.L.get_value()

                # !!! Check again to see if I actually need here.
                update_bn_fn(inputs, targets)
                #update the std and mean in bn layer.

                L = self.Ls[self.idx]                                
                self.L.set_value(np.float32(L * current_factor.get_value()))

                current_loss, current_acc = val_fn(inputs, targets)
#                current_loss = val_fn(np.array(inputs.todense(), dtype=np.float32), np.array(targets, dtype=np.int32))
                
                l_iter = 0
                if self.adaptive: 
                    while True:
                        loss_next, sq_sum = L_fn(inputs, targets)
                        if loss_next <= current_loss - 0.5 * sq_sum / self.L.get_value():
                            break
                        else:
                            self.L.set_value(self.L.get_value() * 2)

                        l_iter += 1

                if EXTRA_INFO:
                    print >>flog, "No. of batch:",train_batches
                    #Each iteration here decrease learning rate by half 
                    flog.write("Iteration of L:{:.2f}\n".format(l_iter))
                    print >>flog, "learning_rate: ", 1. / self.L.get_value()

                #update w
                train_err += train_w(inputs, targets)
                train_acc += current_acc

                self.Ls[self.idx] = self.L.get_value()
                train_batches += 1
            
            val_err = 0
            val_acc = 0
            val_batches = 0
            for i, batch in enumerate(iterate_minibatches(X_val, Y_val, batch_size, shuffle=False)):
                inputs, targets = batch
                current_err, current_acc = val_fn(inputs, targets)
                val_err += current_err
                val_acc += current_acc
                val_batches += 1

            test_err = 0
            test_acc = 0
            test_batches = 0
            for i, batch in enumerate(iterate_minibatches(X_test, y_test, batch_size, shuffle=False)):
                inputs, targets = batch
                current_err, current_acc = val_fn(inputs, targets)
                test_err += current_err
                test_acc += current_acc
                test_batches += 1

            times.append(time.time() - t)
            epoch_times.append(time.time()-start_time)
            
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, n_epochs, time.time() - t))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  train accuracy:\t\t{:.6f}".format(train_acc / train_batches))
            print("  validation accuracy:\t\t{:.6f}".format(val_acc / val_batches))
            print("  test accuracy:\t\t{:.6f}\n".format(test_acc / test_batches))

            train_error.append(train_err / train_batches)
            validation_error.append((val_err / val_batches, self.counted_gradient.get_value()))
            acc_train.append(train_acc / train_batches)
            acc_val.append(val_acc / val_batches)
            test_error.append(test_err / test_batches)
            acc_test.append(test_acc / test_batches)
            
#            if X_val is not None:
#                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

        print("Average time per epoch \t {:.3f}".format(np.mean(times)))
        return train_error, validation_error, acc_train, acc_val, acc_test, test_error, epoch_times


    def make_updates(self, loss, params):

        mu_updates = self.make_mu_updates(loss, params)
        w_updates = self.make_w_updates(loss, params)
   
        return w_updates, mu_updates

    def make_mu_updates(self, loss, params):
        mu_updates = OrderedDict()

        grads = theano.grad(loss, params)

        self.mu = []
        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)

            mu_updates[self.counted_gradient] = self.counted_gradient + 1

            mu = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            mu_updates[mu] = mu + grad
            self.mu.append(mu)

        return mu_updates

    def make_w_updates(self, loss, params):
        w_updates = OrderedDict()
        
        params_tilde = [theano.shared(x.get_value()) for x in params] 
        loss_tilde = theano.clone(loss, replace=zip(params, params_tilde))

        grads = theano.grad(loss, params)
        grads_tilde = theano.grad(loss_tilde, params_tilde)

        it_num = theano.shared(np.cast['int16'](0))
        it = it_num + 1

        for param, grad, mu, param_tilde, grad_tilde in zip(params, grads, self.mu, params_tilde, grads_tilde):
#            new_param = param - self.learning_rate * (grad - grad_tilde + mu)

            new_param = param - (1. / self.L) * (grad - grad_tilde + mu)
            w_updates[param] = new_param
            w_updates[param_tilde] = ifelse(T.eq(it % self.m, 0), new_param, param_tilde)
            
            w_updates[self.counted_gradient] = self.counted_gradient + 2
        
        if self.adaptive:
            w_updates[self.L] = self.L / 2

        self.it_num = it_num
        
        w_updates[it_num] = it
        return w_updates

    def make_L_fn(self, loss, params):
        grads = theano.grad(loss, params)

        params_next = [x - 1. / self.L * g for x, g in zip(params, grads)]
        loss_next = theano.clone(loss, replace=zip(params, params_next))
        sq_sum = sum((g**2).sum() for g in grads)

        return theano.function([self.input_var, self.target_var], [loss_next, sq_sum])

    def iterate_minibatches(self, inputs, targets, batchsize):
        assert inputs.shape[0] == len(targets)
        
        indices = np.arange(inputs.shape[0] / batchsize)
        
        lipschitz_prob = self.get_prob()

        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            [idx] = np.random.choice(indices, size=1, p=lipschitz_prob)
            excerpt = np.arange(inputs.shape[0])[idx * batchsize : (idx + 1) * batchsize]
            self.idx = idx
            yield inputs[excerpt], targets[excerpt]

    def get_prob(self):
#        s = sum(L.get_value() for L in self.Ls)
#        return [L.get_value() / s for L in self.Ls]
        s = sum(self.Ls)
        return [L / s for L in self.Ls]
