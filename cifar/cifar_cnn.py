import sys
import os
import matplotlib.pylab as py
import numpy as np
import pickle,gzip
import time

#Import Theano Packages
import theano
import theano.tensor as T

#Import Lasagne Packages
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
# from lasagne.layers.quantize import compute_grads,clipping_scaling,train,binarization
from lasagne.layers import batch_norm

## ImageneDataGenerator module is added to my fork of Lasagne, to allow for realtime data augmentation
from keras.preprocessing.image import ImageDataGenerator
# Define a Neural Network using Lasagne APIs

import my_bn_layer # Define DBN, e.g., DBN1
import my_bn_layer2 # Define DBN2, e.g. ,1/m^2 MA
import my_bn_layer_const # Define for const alpha (actually we can just use the my_bn_layer)

#input area
NUM_EPOCHS = 2

### no efect
NUM_HIDDEN_UNITS = 100 # no effect
BNALG = 'const1'
OUTPUT_DATA_PATH = 'data_nips_rebuttal/'
MODEL = 'mlpbn'
GRADIENT = 'adagrad'
LR_START = 0.01

bnalg_const_dict = {
"const1":     1.0, 
"const075":   0.75, 
"const05":    0.5, 
"const025":   0.25, 
"const01":    0.1, 
"const001":   0.01, 
"const0001":  0.001, 
"const0":     0.0, }    


# We will build a reasonably complex Convolutional neural network incorporating all the modern innovations such as:
    # a. Batch Norm [See: http://arxiv.org/abs/1502.03167], 
    #b. Dropout [See: http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf] and 
    #c. Intelligent Weight Initialization [See: http://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html]
def batch_norm_adopted(layer,bnalg):
    print "Inside the batch_norm_adopted(), the bnalg is:", bnalg

    if bnalg == 'original':
        return batch_norm(layer)
    elif bnalg == 'dbn':
        return my_bn_layer.my_batch_norm(layer)
    elif bnalg == 'dbn2':
        return my_bn_layer2.my_batch_norm(layer)
    elif 'const' in bnalg:
        if bnalg not in bnalg_const_dict:
            print("Incorrect bnalg method. Can't find in predefined dictionary.")
            return 
        else:
            the_alpha = bnalg_const_dict[bnalg]
            return my_bn_layer_const.my_batch_norm(layer,alpha = the_alpha)

def cnn_network(input_var, bnalg = BNALG,input_channels=3,input_img_size=(32,32),num_classes=10):
    #Choose hidden-node nonlinearity as ReLU function
    nonlinearity=lasagne.nonlinearities.rectify
    #Create an empty dictionary that we will populate with the network layers
    net={}
    #Create an input layer to read input images
    net['l_in']=lasagne.layers.InputLayer((None,input_channels,input_img_size[0],input_img_size[1]),input_var=input_var)
    
    #Create the first convolutional layer with 32 channels, and filter kernel of size 3x3, witg padding such that 
    #output image size is same as the input image size
    #The output convolution layer is passed through a batch normalization layer
    #We also use He initialization to initialize the weights of the convolution filter

    net[0]=batch_norm_adopted(lasagne.layers.Conv2DLayer(net['l_in'],32,3,W=lasagne.init.HeUniform(),pad='same',nonlinearity=nonlinearity), bnalg)
    
    #Add another convolution layer
    net[1]=batch_norm_adopted(lasagne.layers.Conv2DLayer(net[0],32,3,W=lasagne.init.HeUniform(),pad='same',nonlinearity=nonlinearity), bnalg)
    
    #Add max pooling layer.. which reduces the output image size by half
    net[2]=lasagne.layers.MaxPool2DLayer(net[1],2)
    
    #Add a dropout layer.. which serves to regularize the training
    net[3]=lasagne.layers.DropoutLayer(net[2],p=.25)
    
    #Add yet another convolution layer, this time with 64 channels.
    net[4]=batch_norm_adopted(lasagne.layers.Conv2DLayer(net[3],64,3,W=lasagne.init.HeUniform(),pad='same',nonlinearity=nonlinearity), bnalg)
    
    #Repeat with another convolution layer
    net[5]=batch_norm_adopted(lasagne.layers.Conv2DLayer(net[4],64,3,W=lasagne.init.HeUniform(),pad='same',nonlinearity=nonlinearity), bnalg)
    
    #Second stage of max pooling, followed by drop out
    net[6]=lasagne.layers.MaxPool2DLayer(net[5],2)
    net[7]=lasagne.layers.DropoutLayer(net[6],p=.25)
   
    #Add a fully connected layer with 512 hidden nodes with ReLU nonlinearity, followed by a drop out layer
    net[8]=batch_norm_adopted(lasagne.layers.DenseLayer(net[7],num_units=512,nonlinearity=lasagne.nonlinearities.rectify), bnalg)
    net[9]=lasagne.layers.DropoutLayer(net[8],p=.5)
    
    #Finally add the output probability layer to classify the input image into one of 10 classes
    net['l_out']=lasagne.layers.DenseLayer(net[9],num_units=num_classes,nonlinearity=lasagne.nonlinearities.softmax)
   
    return net

# Batch Training Module
def batch_train(datagen,f_train,f_val,output_layer,lr_start,lr_decay,N_train_batches=5,mini_batch_size=32,img_dim=(3,32,32),\
    epochs=10,test_interval=1,data_dir='/Users/sachintalathi/Work/Python/Data/cifar-10-batches-py',\
    data_augment_bool=False,train_bool=True):

    # Input:
        # Datagenerator.. Python generator function for loading input data and performing series of data augmentations
        # f_train: Theano function to compute loss on training data and labels and perform gradient weight updates
        # f_train: Theano function to compute loss on testing (validation) data and labels
        # lr_start: Starting value for learning rate
        # lr_decay: Learning rate decay factor
        # N_train_batches: Number of training data batches: 5
        # Training mini batch size: 32
        # img_dim: (Num_Channels, Img_X, Img_Y)=(3,32,32) --> For Cifar-10 data
        # epochs: Number of training epochs
        # test_interval: The interval in units of epochs, when test scores will be comptued
        # data_dir: Path to Cifar-10 benchmark training and testing data
        # data_augment_bool: Boolean to determine whether data augmentations is performed while training
        # train_bool: Boolean to determine whether to train the network or simply compute test scores
    
    #Module to compute Training Data Mean and Std. Dev
    def Get_Data_Stats(data_dir,img_dim,N_train_batches):
        s=np.zeros((np.prod(img_dim),))
        sq=np.zeros((np.prod(img_dim),))
        for ind in range(N_train_batches):
            D=pickle.load(open('%s/data_batch_%d'%(data_dir,ind+1)))
            data=D['data'].astype('float32')
            s+=np.sum(data,axis=0)
            sq+=np.sum(data*data,axis=0)

        Mean=1.0*s/50000
        Sq_Mean=1.0*sq/50000
        Mean_Sq=Mean*Mean
        Std=np.sqrt(Sq_Mean-Mean_Sq)
        return Mean,Std
    
    # Generator function to read training data
    def batch_gen(X,y,N):
        while True:
            idx=np.random.choice(len(y),N)
            yield X[idx].astype('float32'),y[idx].astype('float32')
    
    # Module to perform gradient descent training on input data batch
    def train_on_batch(batch_index,data_dir,Data_Mean,Data_Std,data_augment_bool,img_dim,mini_batch_size,f_train,LR):
        #Load Data per data-batch
        D=pickle.load(open('%s/data_batch_%d'%(data_dir,batch_index+1)))
        assert 'data' in D.keys()
        data=D['data'].astype('float32')
        #I am not performing data normalization in this example, but can be done as follows:
        #data=(data-Data_Mean)/Data_Std
        #Scale the input data to be between 0 and 1
        data=data/255
        data = data.reshape(data.shape[0], img_dim[0], img_dim[1], img_dim[2])
        assert 'labels' in D.keys()
        labels=np.array(D['labels'])
        train_loss_per_batch=0
        train_err_per_batch=0
        
        # Perform data augmentation using the datagen generator function
        if data_augment_bool:
            train_batches=datagen.flow(data,labels,mini_batch_size) ### Generates data augmentation on the fly
        else:
            train_batches=batch_gen(data,labels,mini_batch_size) ### No data augmentation applied
        N_mini_batches=len(labels)//mini_batch_size
        
        #Perform training on data batch
        for mini_batch in range(N_mini_batches):
            X,y=next(train_batches)
            loss,err=f_train(X,y,LR)
            train_loss_per_batch+=loss
            train_err_per_batch+=err
        train_loss_per_batch=train_loss_per_batch/N_mini_batches
        train_err_per_batch=train_err_per_batch/N_mini_batches
        return train_loss_per_batch,train_err_per_batch
    
    # Module to compute scores on test data batch
    #The only difference here is that we use theano function f_val instead of f_train above
    def val_on_batch(data_dir,img_dim,mini_batch_size,f_val):
        val_loss=0
        val_err=0    
        D_val=pickle.load(open('%s/test_batch'%(data_dir)))
        assert 'data' in D_val.keys()
        data=D_val['data'].astype('float32')
        #data=(data-Data_Mean)/Data_Std
        data=data/255
        data=data.reshape(data.shape[0], img_dim[0], img_dim[1], img_dim[2])
        assert 'labels' in D_val.keys()
        labels=np.array(D_val['labels'])
        val_batches=batch_gen(data,labels,mini_batch_size)
        N_val_batches=len(labels)//mini_batch_size
        for _ in range(N_val_batches):
            X,y=next(val_batches)
            loss,err=f_val(X,y)
            val_loss+=loss
            val_err+=err
        val_loss=val_loss/N_val_batches
        val_err=val_err/N_val_batches
        return val_loss,val_err

    per_epoch_train_stats=[];per_epoch_val_stats=[]
    per_epoch_params=[]
    Data_Mean,Data_Std=Get_Data_Stats(data_dir,img_dim,N_train_batches)
    
    #Now that all the necessary modules are defined, we are ready to perform training
    
    print('Running Epochs')
    LR=lr_start
    for epoch in range(epochs):
        if train_bool:
            train_loss=0
            train_err=0
            for ind in range(N_train_batches):
                tic=time.clock()
                tlpb,tapb=train_on_batch(ind,data_dir,Data_Mean,Data_Std,data_augment_bool,img_dim,mini_batch_size,f_train,LR)
                toc=time.clock()
                print ('Epoch %d Data_Batch (Time) %d (%0.03f s) Learning_Rate %0.04f Train Loss (Error)\
                    %.03f (%.03f)'%(epoch,ind,toc-tic,LR,tlpb,tapb))
                train_loss+=tlpb
                train_err+=tapb
            train_loss=train_loss/N_train_batches
            train_err=train_err/N_train_batches
            per_epoch_train_stats.append([epoch,train_loss,train_err])

            bn_params =    lasagne.layers.get_all_param_values(output_layer, trainable=False)

            this_mu     = np.copy(bn_params[0])
            this_lambda = np.copy(bn_params[1])
            per_epoch_params.append([this_mu,this_lambda])

            # all_mu  = [bn_params[0], bn_params[2],bn_params[4],bn_params[6],bn_params[8]]
            # all_mu  = bn_params[0]
            # all_std = [bn_params[1], bn_params[3],bn_params[5],bn_params[7],bn_params[9]]
            # all_std = bn_params[1]
            # per_epoch_params.append([all_mu,all_std])


        if (epoch+1)%test_interval==0:
            val_loss,val_err=val_on_batch(data_dir,img_dim,mini_batch_size,f_val)

            per_epoch_val_stats.append([epoch,val_loss,val_err])
            print ('Epoch  (Time) %d (%0.03f s) Learning_Rate %0.04f Test Loss (Error)\
                %.03f (%.03f)'%(epoch,toc-tic,LR,val_loss,val_err))
        
        if epoch%10 ==0 and gradient=="sgd":
            LR*=0.9

    #Return the performance Stats after training is complete
    return per_epoch_train_stats,per_epoch_val_stats,per_epoch_params


def main(model=MODEL,gradient = GRADIENT, num_epochs=NUM_EPOCHS, num_hidden_units = NUM_HIDDEN_UNITS, bnalg = BNALG, lr_start = LR_START):

    # Set the Initial Learning Rate; the Final Learning Rate and the number of training epochs
    LR_start= lr_start
    LR_fin = 0.01
    epochs=num_epochs
    # LR_decay = (LR_fin/LR_start)**(1./epochs)
    LR_decay = 1

    print("Generating the ImageDataGenerator")
    #Define the Image Data Generator, which is used for real-time data augmentation while training
    datagen = ImageDataGenerator(
              featurewise_center=False,  # set input mean to 0 over the dataset
              samplewise_center=False,  # set each sample mean to 0
              featurewise_std_normalization=False,  # divide inputs by std of the dataset
              samplewise_std_normalization=False,  # divide each input by its std
              zca_whitening=False,  # apply ZCA whitening
              rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
              width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
              height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
              horizontal_flip=True,  # randomly flip images
              vertical_flip=False)  # randomly flip images

    #Define Theano tensor variables for input, the labels and the learning rate
    input=T.tensor4('input')
    target=T.ivector('target')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    #Define the Network
    print("Generating the cnn network")
    net=cnn_network(input, bnalg)

    #Define Training Output Variables

    print("Compiling the functions")

    train_output=lasagne.layers.get_output(net['l_out'],input,deterministic=False) ## Get the class probabilities
    train_pred=train_output.argmax(-1) ## Get the predicted class label
    train_loss=T.mean(lasagne.objectives.categorical_crossentropy(train_output,target)) #Using Cross-Entropy Loss
    train_err=T.mean(T.neq(T.argmax(train_output,axis=1), target),dtype=theano.config.floatX) #Compute the mean training precdiction error

    # Define Validation Output Variables

    val_output=lasagne.layers.get_output(net['l_out'],input,deterministic=True)
    val_loss=T.mean(lasagne.objectives.categorical_crossentropy(val_output,target))
    val_err = T.mean(T.neq(T.argmax(val_output,axis=1), target),dtype=theano.config.floatX)
    val_pred=val_output.argmax(-1)

    # Set L2 regularization coefficient
    layers={}
    for k in net.keys():
        layers[net[k]]=0.0005

    l2_penalty = regularize_layer_params_weighted(layers, l2)
    train_loss=train_loss+l2_penalty

    #Define the Gradient Update Rule
    print("Compiling the functions: extract params")

    params = lasagne.layers.get_all_params(net['l_out'], trainable=True) #Get list of all trainable network parameters
    # bnparams = lasagne.layers.get_all_params(net['l_out'], trainable=False) #Get list of all BN untrainable network parameters

    if gradient == "adagrad":
        updates = lasagne.updates.adagrad(loss_or_grads=train_loss, params=params, learning_rate=LR) 
        ## Use Adagrad Gradient Descent Learning Algorithm
    elif gradient == "rmsprop":
        updates = lasagne.updates.rmsprop(loss_or_grads=train_loss, params=params, learning_rate=LR) 
    elif gradient == "sgd":
        updates = lasagne.updates.sgd(loss_or_grads=train_loss, params=params, learning_rate=LR) 
    else:
        print("Invalid gradient name")

    #Define Theano Functions for Training and Validation

    #Theano Function for Training
    print("Compiling the functions: define train function")

    f_train=theano.function([input,target,LR],[train_loss,train_err],updates=updates,allow_input_downcast=True)

    # Theano Function for Validation
    
    print("Compiling the functions: define val function")
    f_val=theano.function([input,target],[val_loss,val_err],allow_input_downcast=True)

    # f_get_params=theano.function([LR],[bnparams],allow_input_downcast=True)

    #Begin Training
    print("Beging Training")
    train_stats,val_stats,per_epoch_params=batch_train(datagen,f_train,f_val,net['l_out'],LR_start,LR_decay,epochs=epochs,\
        data_dir="../../data/cifar-10-batches-py/",train_bool=True)

    #output data
    list_epoch      = [i[0] for i in val_stats]
    list_val_loss   = [i[1] for i in val_stats]
    list_val_err    = [i[2] for i in val_stats]
    list_val_acc    = [1-i  for i in list_val_err]

    epoch_mu        = [i[0] for i in per_epoch_params]
    epoch_lambda    = [i[1] for i in per_epoch_params]
    
    # epoch_params_mu = [i[0] for i in epoch_params]
    # epoch_params_std= [i[1] for i in epoch_params]

    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(num_epochs)+"_"+bnalg+"_"+"epoch.txt",list_epoch)
    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(num_epochs)+"_"+bnalg+"_"+"loss_val.txt",list_val_loss)
    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(num_epochs)+"_"+bnalg+"_"+"acc_val.txt",list_val_acc)
    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(num_epochs)+"_"+bnalg+"_"+"err_val.txt",list_val_err)
    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(num_epochs)+"_"+bnalg+"_"+"params_mu.txt",epoch_mu)
    np.savetxt(OUTPUT_DATA_PATH+model+"_"+gradient+"_"+str(num_epochs)+"_"+bnalg+"_"+"params_std.txt",epoch_lambda)
    print ("Data saved...")


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv) or ('help' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL] [GRADIENT] [NUM_EPOCHS]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'mlpbn: for an MLP with batch Normalization")
        print("GRADIENT: 'sgd', 'svrg'")
        print("NUM_EPOCHS: ")
        print("NUM_HIDDEN_UNITS: (No effect)")
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
        if len(sys.argv) > 6:
            kwargs['lr_start'] = float(sys.argv[6])
        main(**kwargs)



#Plot the Evolution of Training and Testing
# import matplotlib.pylab as py
# py.ion()
# py.figure();py.plot(np.array(train_stats)[:,2]);py.hold('on');py.plot(np.array(val_stats)[:,2]);






