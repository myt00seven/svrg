import numpy as np
import matplotlib.pyplot as plt
import seaborn

def plot(params):

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'darkgreen', 'lime']
    for i, model in enumerate(params):
        with np.load('models/model_%s_val_error.npz' % model) as f:
            val_err = f['arr_0']
        
#        if model.find('nesterov') != -1:
#            style = '--'
#        elif model.find('div') != -1:
#            style = '-.'
#        else:
#            style = '-'
        style = '-'

        if type(val_err[0]) == np.ndarray:
            y, x = zip(*val_err)
            print y
            plt.plot(x, y, label=model, linestyle=style, color=colors[i])
        else:
            plt.plot(range(len(val_err)), val_err, label=model, linestyle=style, color=colors[i])

#params = [
#    'custom_momentum-1.0-0.9',
#    'custom_momentum-0.1-0.9',
#    'custom_momentum-0.01-0.9',
#    'custom_momentum-0.001-0.9',
#    'custom_momentum-0.1-0.5',
#    'custom_momentum-0.1-0.1',
#]

params = [
    'custom_momentum-1.0divk**1.0-0.9',
    'custom_momentum-1.0divk**0.75-0.9',
    'custom_momentum-1.0divk**0.5-0.9',
    'custom_nesterov_momentum-1.0divk**1.0-0.9',
    'custom_nesterov_momentum-1.0divk**0.75-0.9',
    'custom_nesterov_momentum-1.0divk**0.5-0.9',
    'custom_momentum-1.0-0.9'
]

#params = [
#    'custom_adagrad_0.01',
#    'custom_adagrad_0.1',
#    'custom_adagrad_1.0',
#    'custom_adagrad_10.0'
#]

#params = [
#    'custom_rmsprop_0.01-0.9',
#    'custom_rmsprop_0.01-0.6',
#    'custom_rmsprop_0.01-0.3',
#    'custom_rmsprop_0.01-0.1'
#]

#params = [
#    'custom_adam_0.01_0.9_0.999',
#    'custom_adam_0.01_0.5_0.999',
#    'custom_adam_0.01_0.1_0.999',
#                               
#    'custom_adam_0.01_0.9_0.5',
#    'custom_adam_0.01_0.9_0.1',
#
#    'custom_adam_0.1_0.9_0.999',
#    'custom_adam_1.0_0.9_0.999',
#    'custom_adam_10.0_0.9_0.999',
#]

#params = [
#
#    'custom_adam_0.01_0.9_0.999',
#    'custom_adagrad_0.1',
#    'custom_rmsprop_0.01-0.9',
#    'custom_momentum-1.0divk**0.5-0.9',
#    'custom_momentum-1.0-0.9',
#]

params = [
#    'svrg_100.0m_300',
#    'momentum_1.0_0.9_300',
 #   'svrg_test',
#    'adam_test',
    'svrg_test_faster100epochs',
    'adam_test_faster100epochs',
    'sdg_test',
    'sdg_test_long',
#    'sdg_test_long_nomomentum',
    'svrg_testing',
    'svrg_testing_nonadaptive',
    'svrg_testing_nonadaptive_withprob',
#    'svrg_test_100',
#    'adam_test_100',
]

#params = [
#    'svrg_news_data',
#    'adam_news_data',
#    'sgdm_news_data'
#]

plot(params)

plt.title('Validation error/epoch')    
plt.legend()
plt.show()

