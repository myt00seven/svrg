import sys
sys.path.append('./lib')
import theano
theano.config.on_unused_input = 'warn'
import theano.tensor as T

import numpy as np

from layers import DataLayer, ConvPoolLayer, DropoutLayer, FCLayer, SoftmaxLayer

class VggNet(object):

    def __init__(self, config):

        self.config = config

        batch_size = config['batch_size']
        flag_datalayer = config['use_data_layer']
        lib_conv = config['lib_conv']

        # ##################### BUILD NETWORK ##########################
        # allocate symbolic variables for the data
        # 'rand' is a random array used for random cropping/mirroring of data
        x = T.ftensor4('x')
        y = T.lvector('y')
        rand = T.fvector('rand')

        print '... building the model'
        self.layers = []
        params = []
        weight_types = []

        if flag_datalayer:
            data_layer = DataLayer(input=x, image_shape=(3, 256, 256,
                                                         batch_size),
                                   cropsize=224, rand=rand, mirror=True,
                                   flag_rand=config['rand_crop'])

            layer1_input = data_layer.output
        else:
            layer1_input = x

        convpool_layer1_1 = ConvPoolLayer(input=layer1_input,
                                        image_shape=(3, 224, 224, batch_size), 
                                        filter_shape=(3, 3, 3, 64), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer1_1)
        params += convpool_layer1_1.params
        weight_types += convpool_layer1_1.weight_type

        convpool_layer1_2 = ConvPoolLayer(input=convpool_layer1_1.output,
                                        image_shape=(64, 224, 224, batch_size), 
                                        filter_shape=(64, 3, 3, 64), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=2, poolstride=2, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer1_2)
        params += convpool_layer1_2.params
        weight_types += convpool_layer1_2.weight_type

        convpool_layer2_1 = ConvPoolLayer(input=convpool_layer1_2.output,
                                        image_shape=(64, 112, 112, batch_size), 
                                        filter_shape=(64, 3, 3, 128), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer2_1)
        params += convpool_layer2_1.params
        weight_types += convpool_layer2_1.weight_type

        convpool_layer2_2 = ConvPoolLayer(input=convpool_layer2_1.output,
                                        image_shape=(128, 112, 112, batch_size), 
                                        filter_shape=(128, 3, 3, 128), 
                                        convstride=1, padsize=1, group=2, 
                                        poolsize=2, poolstride=2, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer2_2)
        params += convpool_layer2_2.params
        weight_types += convpool_layer2_2.weight_type


        convpool_layer3_1 = ConvPoolLayer(input=convpool_layer2_2.output,
                                        image_shape=(128, 56, 56, batch_size), 
                                        filter_shape=(128, 3, 3, 256), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer3_1)
        params += convpool_layer3_1.params
        weight_types += convpool_layer3_1.weight_type

        convpool_layer3_2 = ConvPoolLayer(input=convpool_layer3_1.output,
                                        image_shape=(256, 56, 56, batch_size), 
                                        filter_shape=(256, 3, 3, 256), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer3_2)
        params += convpool_layer3_2.params
        weight_types += convpool_layer3_2.weight_types

        convpool_layer3_3 = ConvPoolLayer(input=convpool_layer3_2.output,
                                        image_shape=(256, 56, 56, batch_size), 
                                        filter_shape=(256, 3, 3, 256), 
                                        convstride=1, padsize=1, group=2, 
                                        poolsize=2, poolstride=2, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer3_3)
        params += convpool_layer3_3.params
        weight_types += convpool_layer3_3.weight_types

        convpool_layer4_1 = ConvPoolLayer(input=convpool_layer3_3.output,
                                        image_shape=(256, 28, 28, batch_size), 
                                        filter_shape=(256, 3, 3, 512), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer4_1)
        params += convpool_layer4_1.params
        weight_types += convpool_layer4_1.weight_type

        convpool_layer4_2 = ConvPoolLayer(input=convpool_layer4_1.output,
                                        image_shape=(512, 28, 28, batch_size), 
                                        filter_shape=(512, 3, 3, 512), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer4_2)
        params += convpool_layer4_2.params
        weight_types += convpool_layer4_2.weight_types

        convpool_layer4_3 = ConvPoolLayer(input=convpool_layer4_2.output,
                                        image_shape=(512, 28, 28, batch_size), 
                                        filter_shape=(512, 3, 3, 512), 
                                        convstride=1, padsize=1, group=2, 
                                        poolsize=2, poolstride=2, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer4_3)
        params += convpool_layer4_3.params
        weight_types += convpool_layer4_3.weight_types

        convpool_layer5_1 = ConvPoolLayer(input=convpool_layer4_3.output,
                                        image_shape=(512, 14, 14, batch_size), 
                                        filter_shape=(512, 3, 3, 512), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer5_1)
        params += convpool_layer5_1.params
        weight_types += convpool_layer5_1.weight_type

        convpool_layer5_2 = ConvPoolLayer(input=convpool_layer5_1.output,
                                        image_shape=(512, 14, 14, batch_size), 
                                        filter_shape=(512, 3, 3, 512), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer5_2)
        params += convpool_layer5_2.params
        weight_types += convpool_layer5_2.weight_types

        convpool_layer5_3 = ConvPoolLayer(input=convpool_layer5_2.output,
                                        image_shape=(512, 14, 14, batch_size), 
                                        filter_shape=(512, 3, 3, 512), 
                                        convstride=1, padsize=1, group=2, 
                                        poolsize=2, poolstride=2, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer5_3)
        params += convpool_layer5_3.params
        weight_types += convpool_layer5_3.weight_types

        fc_layer6_input = T.flatten(
            convpool_layer5.output.dimshuffle(3, 0, 1, 2), 2)
        fc_layer6 = FCLayer(input=fc_layer6_input, n_in=25088, n_out=4096)
        self.layers.append(fc_layer6)
        params += fc_layer6.params
        weight_types += fc_layer6.weight_type

        dropout_layer6 = DropoutLayer(fc_layer6.output, n_in=4096, n_out=4096)

        fc_layer7 = FCLayer(input=dropout_layer6.output, n_in=4096, n_out=4096)
        self.layers.append(fc_layer7)
        params += fc_layer7.params
        weight_types += fc_layer7.weight_type

        dropout_layer7 = DropoutLayer(fc_layer7.output, n_in=4096, n_out=4096)

        softmax_layer8 = SoftmaxLayer(
            input=dropout_layer7.output, n_in=4096, n_out=1000)
        self.layers.append(softmax_layer8)
        params += softmax_layer8.params
        weight_types += softmax_layer8.weight_type

        # #################### NETWORK BUILT #######################

        self.cost = softmax_layer8.negative_log_likelihood(y)
        self.errors = softmax_layer8.errors(y)
        self.errors_top_5 = softmax_layer8.errors_top_x(y, 5)
        self.params = params
        self.x = x
        self.y = y
        self.rand = rand
        self.weight_types = weight_types
        self.batch_size = batch_size


def compile_models(model, config, flag_top_5=False):

    x = model.x
    y = model.y
    rand = model.rand
    weight_types = model.weight_types

    cost = model.cost
    params = model.params
    errors = model.errors
    errors_top_5 = model.errors_top_5
    batch_size = model.batch_size

    mu = config['momentum']
    eta = config['weight_decay']

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = []

    learning_rate = theano.shared(np.float32(config['learning_rate']))
    lr = T.scalar('lr')  # symbolic learning rate

    if config['use_data_layer']:
        raw_size = 256
    else:
        raw_size = 227

    shared_x = theano.shared(np.zeros((3, raw_size, raw_size,
                                       batch_size),
                                      dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(np.zeros((batch_size,), dtype=int),
                             borrow=True)

    rand_arr = theano.shared(np.zeros(3, dtype=theano.config.floatX),
                             borrow=True)

    vels = [theano.shared(param_i.get_value() * 0.)
            for param_i in params]

    if config['use_momentum']:

        assert len(weight_types) == len(params)

        for param_i, grad_i, vel_i, weight_type in \
                zip(params, grads, vels, weight_types):

            if weight_type == 'W':
                real_grad = grad_i + eta * param_i
                real_lr = lr
            elif weight_type == 'b':
                real_grad = grad_i
                real_lr = 2. * lr
            else:
                raise TypeError("Weight Type Error")

            if config['use_nesterov_momentum']:
                vel_i_next = mu ** 2 * vel_i - (1 + mu) * real_lr * real_grad
            else:
                vel_i_next = mu * vel_i - real_lr * real_grad

            updates.append((vel_i, vel_i_next))
            updates.append((param_i, param_i + vel_i_next))

    else:
        for param_i, grad_i, weight_type in zip(params, grads, weight_types):
            if weight_type == 'W':
                updates.append((param_i,
                                param_i - lr * grad_i - eta * lr * param_i))
            elif weight_type == 'b':
                updates.append((param_i, param_i - 2 * lr * grad_i))
            else:
                raise TypeError("Weight Type Error")

    # Define Theano Functions

    train_model = theano.function([], cost, updates=updates,
                                  givens=[(x, shared_x), (y, shared_y),
                                          (lr, learning_rate),
                                          (rand, rand_arr)])

    validate_outputs = [cost, errors]
    if flag_top_5:
        validate_outputs.append(errors_top_5)

    validate_model = theano.function([], validate_outputs,
                                     givens=[(x, shared_x), (y, shared_y),
                                             (rand, rand_arr)])

    train_error = theano.function(
        [], errors, givens=[(x, shared_x), (y, shared_y), (rand, rand_arr)])

    return (train_model, validate_model, train_error,
            learning_rate, shared_x, shared_y, rand_arr, vels)
