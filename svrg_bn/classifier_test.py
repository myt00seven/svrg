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
IF_SWITCH = 0
IF_DATA_SHAKE = 0 # give 1 -1 1 -1 data

# as the training set of MNIST is 50000, set the batch size to 100 means it taks 500 batches to go through the entire training set

def main(model=MODEL,gradient = GRADIENT, n_epochs=NUM_EPOCHS, n_hidden = NUM_HIDDEN_UNITS, if_switch = IF_SWITCH):

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(IF_DATA_SHAKE)

    # n_epochs = 1000
    # n_hidden = 500

    objective = lasagne.objectives.categorical_crossentropy

    models = {}
    l_r = theano.shared(np.array(0.01, dtype="float32")) 
    ada_factor = theano.shared(np.array(1.0, dtype="float32")) 
    if gradient == "svrg" or gradient == "all":
        models.update({ 'svrg': (custom_svrg1, {'learning_rate': 0.01, 'm': 50, 'adaptive': False, 'adaptive_half_life_period':20, 'ada_factor':ada_factor}) })
    if gradient == "streaming" or gradient == "all": # It is StreamingSVRG
        models.update({ 'streaming': (custom_streaming_svrg1, {'learning_rate': 0.01, 'm': 50, 'k_s_0': 1.0, 'k_s_ratio':1.10, 'adaptive': False, 'adaptive_half_life_period':20, 'ada_factor':ada_factor}) })
        #k_s is the ratio of how many batches are used in this iteration of StreamingSVRG
    if gradient == "adagrad" or gradient == "all":        
        models.update( { 'adagrad': (custom_adagrad, {'learning_rate': l_r, 'eps': 1.0e-8, 'adaptive': False, 'adaptive_half_life_period':20, 'ada_factor':ada_factor}) })

    model_adagrad={}
    model_adagrad.update({
        'adagrad': (custom_adagrad, {'learning_rate': l_r, 'eps': 1.0e-8, 'adaptive': False, 'adaptive_half_life_period':20, 'ada_factor':ada_factor})
    })
    
    
    model_streaming={}
    model_streaming.update({'streaming':(custom_streaming_svrg1, {'learning_rate': 0.01, 'm': 50, 'k_s_0': 1.0, 'k_s_ratio':1.10, 'adaptive': False, 'adaptive_half_life_period':20, 'ada_factor':ada_factor})    })
    

    # print(models.keys())

    # models = {
    #     # 'svrg_classif': (custom_svrg1, {'learning_rate': 0.01, 'm': 50})
    #     # 'adam_classif': (custom_adam, {'learning_rate': 0.01})
    #    # 'adam_classif_dropout': (lasagne.updates.adam, {'learning_rate': 0.01})
    # }

    for model in models.keys():
        update, update_params = models[model]
        set_switch = [0.2, 0.4 ,0.6,0.8]

        # seed = int(np.random.random()*10000000)
        seed = 19921010

        np.random.seed(seed)

        file_seed = open("data/best_result_"+gradient+".txt",'a+')
        file_seed.write("Rand Seed: {:d}\n".format(seed))
        file_seed.close()



        network = neuralclassifier.NeuralClassifier(n_input=X_train.shape[1], n_hidden=n_hidden, n_output=10)

        if if_switch:

            for switch_ratio in set_switch:
            
                network1 = neuralclassifier.NeuralClassifier(n_input=X_train.shape[1], n_hidden=n_hidden, n_output=10)                
                
                for model1 in model_adagrad.keys():
                    update1, update_params1 = model_adagrad[model1] 
                    train_err1, val_err1, acc_train1, acc_val1, acc_test1, test_error1, epoch_times1 = network1.train(X_train, y_train, X_val, y_val, X_test, y_test,
                                               n_epochs=int(n_epochs*switch_ratio), lambd=0.1,
                                               objective=objective, update=update1, batch_size=BATCH_SIZE, gradient=model1,  **update_params1 )

                for model2 in model_streaming.keys():
                    update2, update_params2 = model_streaming[model2]
                    train_err2, val_err2, acc_train2, acc_val2, acc_test2, test_error2, epoch_times2 = network1.train(X_train, y_train, X_val, y_val, X_test, y_test,
                                               n_epochs=int(n_epochs*(1-switch_ratio)), lambd=0.1,
                                               objective=objective, update=update2, batch_size=BATCH_SIZE, gradient=model2,  **update_params2 )

                train_err  = train_err1 + train_err2
                val_err  = val_err1 + val_err2
                acc_train  = acc_train1 + acc_train2
                acc_val  = acc_val1 + acc_val2
                acc_test  = acc_test1 + acc_test2
                test_error  = test_error1 + test_error2            
                time_diff = epoch_times1[int(n_epochs*switch_ratio)-1]
                epoch_times2 = [x+time_diff for x in epoch_times2]
                epoch_times  = epoch_times1  + epoch_times2

                np.savetxt("data/"+"ratio_"+str(switch_ratio)+"_loss_train.txt",train_err)
                np.savetxt("data/"+"ratio_"+str(switch_ratio)+"_loss_val.txt",val_err)            
                np.savetxt("data/"+"ratio_"+str(switch_ratio)+"_loss_test.txt",test_error)

                np.savetxt("data/"+"ratio_"+str(switch_ratio)+"_acc_train.txt",acc_train)
                np.savetxt("data/"+"ratio_"+str(switch_ratio)+"_acc_val.txt",acc_val)
                np.savetxt("data/"+"ratio_"+str(switch_ratio)+"_acc_test.txt",acc_test)
                np.savetxt("data/"+"ratio_"+str(switch_ratio)+"_epoch_times.txt",epoch_times)
            
        else:
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
        kwargs['if_switch'] = int(sys.argv[4])
    if len(sys.argv) > 5:
        kwargs['n_hidden'] = int(sys.argv[5])

    main(**kwargs)    
