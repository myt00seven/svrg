# This is used to draw three comparisons for SGD+BN, SVRG+BN and Streaming SVRG +BN

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg') 
import pylab
import numpy as np
		
import sys

# all the methods are with BN layers!!!


PATH_DATA_adagrad  = "data/"
PATH_DATA_SVRG     = "data/"
PATH_DATA_Stream   = "data/"
PATH_DATA   = "data/"

PATH_FIGURE = "figure_3/"

MAXLENGTH = 199
STARTPOINT = 9
LINEWIDTH = 3

DRAW_COMPARE = False

LOAD_SGD = True
LOAD_SVRG = True

DRAW_MLPBN_ADASGD = True
DRAW_MLPBN_streaming = True
DRAW_MLPBN_SVRG = True

DRAW_BN_PARA = True

Y_LIM_FINE_TUNING = True

# SPEC_L1 = 'bo-'
# SPEC_L1 = 'g^--'
# SPEC_L3 = 'cs:'
# SPEC_L4 = 'r*-.'

SPEC_L1 = 'b-'
SPEC_L2 = 'c:'
SPEC_L3 = 'r-.'
SPEC_L4 = 'g--'

NUM_EPOCHS = 1000

def main(num_epochs=NUM_EPOCHS):

	if DRAW_COMPARE:

		str_epochs = str(num_epochs)

		if LOAD_SGD: mlpbn_ADAsgd_acc_test=			np.loadtxt(PATH_DATA +"_mlpbnTrue_adagrad_acc_test.txt")
		if LOAD_SGD: mlpbn_ADAsgd_acc_train=		np.loadtxt(PATH_DATA +"_mlpbnTrue_adagrad_acc_train.txt")
		if LOAD_SGD: mlpbn_ADAsgd_acc_val=			np.loadtxt(PATH_DATA +"_mlpbnTrue_adagrad_acc_val.txt")
		if LOAD_SGD: mlpbn_ADAsgd_loss_test=		np.loadtxt(PATH_DATA +"_mlpbnTrue_adagrad_loss_test.txt")
		if LOAD_SGD: mlpbn_ADAsgd_loss_train=		np.loadtxt(PATH_DATA +"_mlpbnTrue_adagrad_loss_train.txt")
		if LOAD_SGD: mlpbn_ADAsgd_loss_val=			np.loadtxt(PATH_DATA +"_mlpbnTrue_adagrad_loss_val.txt")
		if LOAD_SGD: mlpbn_ADAsgd_epoch_times=			np.loadtxt(PATH_DATA +"_mlpbnTrue_adagrad_epoch_times.txt")

		if LOAD_SVRG: mlpbn_streaming_acc_train=		np.loadtxt(PATH_DATA_SVRG +"_mlpbnTrue_streaming_acc_train.txt")
		if LOAD_SVRG: mlpbn_streaming_acc_val=			np.loadtxt(PATH_DATA_SVRG +"_mlpbnTrue_streaming_acc_val.txt")
		if LOAD_SVRG: mlpbn_streaming_loss_train=		np.loadtxt(PATH_DATA_SVRG +"_mlpbnTrue_streaming_loss_train.txt")
		if LOAD_SVRG: mlpbn_streaming_loss_val=			np.loadtxt(PATH_DATA_SVRG +"_mlpbnTrue_streaming_loss_val.txt")
		if LOAD_SVRG: mlpbn_streaming_acc_test=			np.loadtxt(PATH_DATA_SVRG +"_mlpbnTrue_streaming_acc_test.txt")
		if LOAD_SVRG: mlpbn_streaming_loss_test=		np.loadtxt(PATH_DATA_SVRG +"_mlpbnTrue_streaming_loss_test.txt")
		if LOAD_SVRG: mlpbn_streaming_epoch_times=		np.loadtxt(PATH_DATA_SVRG +"_mlpbnTrue_streaming_epoch_times.txt")
		
		if LOAD_SVRG: mlpbn_svrg_acc_train=		np.loadtxt(PATH_DATA +"_mlpbnTrue_svrg_acc_train.txt")
		if LOAD_SVRG: mlpbn_svrg_acc_val=		np.loadtxt(PATH_DATA +"_mlpbnTrue_svrg_acc_val.txt")
		if LOAD_SVRG: mlpbn_svrg_loss_train=	np.loadtxt(PATH_DATA +"_mlpbnTrue_svrg_loss_train.txt")
		if LOAD_SVRG: mlpbn_svrg_loss_val=		np.loadtxt(PATH_DATA +"_mlpbnTrue_svrg_loss_val.txt")
		if LOAD_SVRG: mlpbn_svrg_acc_test=		np.loadtxt(PATH_DATA +"_mlpbnTrue_svrg_acc_test.txt")
		if LOAD_SVRG: mlpbn_svrg_loss_test=		np.loadtxt(PATH_DATA +"_mlpbnTrue_svrg_loss_test.txt")
		if LOAD_SVRG: mlpbn_svrg_epoch_times=		np.loadtxt(PATH_DATA +"_mlpbnTrue_svrg_epoch_times.txt")

		# count_mlpbn_ADAsgd = 200
		# count_mlpbn_streaming = 200
		# count_mlpbn_svrg = 200

		if DRAW_MLPBN_ADASGD: 	count_mlpbn_ADAsgd = np.arange(mlpbn_ADAsgd_acc_val.shape[0])+1
		if DRAW_MLPBN_streaming: 	count_mlpbn_streaming = np.arange(mlpbn_streaming_acc_train.shape[0])+1
		if DRAW_MLPBN_SVRG: count_mlpbn_svrg = np.arange(mlpbn_svrg_acc_val.shape[0])+1

		# print mlp_sgd_acc_train

		if (MAXLENGTH>0 or STARTPOINT>0):		# Need add for epoch_times
			if DRAW_MLPBN_ADASGD: 	count_mlpbn_ADAsgd = count_mlpbn_ADAsgd[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_streaming: 	count_mlpbn_streaming = count_mlpbn_streaming[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_SVRG: count_mlpbn_svrg = count_mlpbn_svrg[STARTPOINT:MAXLENGTH+1]		
			
			if DRAW_MLPBN_ADASGD: 	mlpbn_ADAsgd_acc_test = mlpbn_ADAsgd_acc_test[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_streaming: 	mlpbn_streaming_acc_test = mlpbn_streaming_acc_test[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_SVRG: mlpbn_svrg_acc_test = mlpbn_svrg_acc_test[STARTPOINT:MAXLENGTH+1]		
			
			if DRAW_MLPBN_ADASGD: 	mlpbn_ADAsgd_loss_test = mlpbn_ADAsgd_loss_test[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_streaming: 	mlpbn_streaming_loss_test = mlpbn_streaming_loss_test[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_SVRG: mlpbn_svrg_loss_test = mlpbn_svrg_loss_test[STARTPOINT:MAXLENGTH+1]
			
			if DRAW_MLPBN_ADASGD: 	mlpbn_ADAsgd_acc_val = mlpbn_ADAsgd_acc_val[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_streaming: 	mlpbn_streaming_acc_val = mlpbn_streaming_acc_val[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_SVRG: mlpbn_svrg_acc_val = mlpbn_svrg_acc_val[STARTPOINT:MAXLENGTH+1]		
			
			if DRAW_MLPBN_ADASGD: 	mlpbn_ADAsgd_loss_val = mlpbn_ADAsgd_loss_val[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_streaming: 	mlpbn_streaming_loss_val = mlpbn_streaming_loss_val[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_SVRG: mlpbn_svrg_loss_val = mlpbn_svrg_loss_val[STARTPOINT:MAXLENGTH+1]
			
			if DRAW_MLPBN_ADASGD: 	mlpbn_ADAsgd_acc_train = mlpbn_ADAsgd_acc_train[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_streaming: 	mlpbn_streaming_acc_train = mlpbn_streaming_acc_train[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_SVRG: mlpbn_svrg_acc_train = mlpbn_svrg_acc_train[STARTPOINT:MAXLENGTH+1]		
					
			if DRAW_MLPBN_ADASGD: 	mlpbn_ADAsgd_loss_train = mlpbn_ADAsgd_loss_train[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_streaming: 	mlpbn_streaming_loss_train = mlpbn_streaming_loss_train[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_SVRG: mlpbn_svrg_loss_train = mlpbn_svrg_loss_train[STARTPOINT:MAXLENGTH+1]

			if DRAW_MLPBN_ADASGD: 	mlpbn_ADAsgd_epoch_times = mlpbn_ADAsgd_epoch_times[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_streaming: 	mlpbn_streaming_epoch_times = mlpbn_streaming_epoch_times[STARTPOINT:MAXLENGTH+1]
			if DRAW_MLPBN_SVRG: mlpbn_svrg_epoch_times = mlpbn_svrg_epoch_times[STARTPOINT:MAXLENGTH+1]



		#PLOT 
		matplotlib.rcParams.update({'font.size': 16})
		plt.figure(1)
		plt.title('Loss of Validation Set')
		
		if DRAW_MLPBN_ADASGD: 	plt.plot(count_mlpbn_ADAsgd, mlpbn_ADAsgd_loss_val, SPEC_L1 ,label="AdaGrad",  linewidth = LINEWIDTH)
		if DRAW_MLPBN_streaming: 	plt.plot(count_mlpbn_streaming, mlpbn_streaming_loss_val, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
		if DRAW_MLPBN_SVRG:	plt.plot(count_mlpbn_svrg, mlpbn_svrg_loss_val, SPEC_L4 ,label="SVRG",  linewidth = LINEWIDTH)
		plt.xlabel('# Epochs')
		plt.ylabel('Loss')
		plt.legend()
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CroszsModel_Validation_Set_Loss'+'.png',bbox_inches='tight')

		plt.figure(2)
		plt.title('Predict Accuracy of Validation Set')
		
		if DRAW_MLPBN_ADASGD: 	plt.plot(count_mlpbn_ADAsgd, mlpbn_ADAsgd_acc_val, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
		if DRAW_MLPBN_streaming: 	plt.plot(count_mlpbn_streaming, mlpbn_streaming_acc_val, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
		if DRAW_MLPBN_SVRG:	plt.plot(count_mlpbn_svrg, mlpbn_svrg_acc_val, SPEC_L4 ,label="SVRG",  linewidth = LINEWIDTH)
		plt.xlabel('# Epochs')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CrossModel_Validation_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(3)
		plt.title('Loss of Training Set')
		# if Y_LIM_FINE_TUNING:	pylab.ylim([-0.01,0.25])	
		if DRAW_MLPBN_ADASGD: 	plt.plot(count_mlpbn_ADAsgd, mlpbn_ADAsgd_loss_train, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
		if DRAW_MLPBN_streaming: 	plt.plot(count_mlpbn_streaming, mlpbn_streaming_loss_train, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
		if DRAW_MLPBN_SVRG:	plt.plot(count_mlpbn_svrg, mlpbn_svrg_loss_train, SPEC_L4 ,label="SVRG", linewidth = LINEWIDTH)
		plt.xlabel('# Epochs')
		plt.ylabel('Loss')
		plt.legend()
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CrossModel_Training_Set_Loss'+'.png',bbox_inches='tight')

		plt.figure(4)
		plt.title('Predict Accuracy of Training Set')
		if Y_LIM_FINE_TUNING:	pylab.ylim([0.93,1.01])	
		if DRAW_MLPBN_ADASGD: 	plt.plot(count_mlpbn_ADAsgd, mlpbn_ADAsgd_acc_train, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
		if DRAW_MLPBN_streaming: 	plt.plot(count_mlpbn_streaming, mlpbn_streaming_acc_train, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
		if DRAW_MLPBN_SVRG:	plt.plot(count_mlpbn_svrg, mlpbn_svrg_acc_train, SPEC_L4 ,label="SVRG",  linewidth = LINEWIDTH)
		plt.xlabel('# Epochs')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CrossModel_Training_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(5)
		plt.title('Predict Accuracy of Test Set')
		
		if DRAW_MLPBN_ADASGD: plt.plot(count_mlpbn_ADAsgd, mlpbn_ADAsgd_acc_test, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
		if DRAW_MLPBN_streaming: 	plt.plot(count_mlpbn_streaming, mlpbn_streaming_acc_test, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
		if DRAW_MLPBN_SVRG: 	plt.plot(count_mlpbn_svrg, mlpbn_svrg_acc_test, SPEC_L4 ,label="SVRG", linewidth = LINEWIDTH)
		plt.xlabel('# Epochs')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CrossModel_Test_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(6)
		plt.title('Loss of Test Set')
		
		if DRAW_MLPBN_ADASGD: plt.plot(count_mlpbn_ADAsgd, mlpbn_ADAsgd_loss_test, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
		if DRAW_MLPBN_streaming: 	plt.plot(count_mlpbn_streaming, mlpbn_streaming_loss_test, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
		if DRAW_MLPBN_SVRG: 	plt.plot(count_mlpbn_svrg, mlpbn_svrg_loss_test, SPEC_L4 ,label="SVRG", linewidth = LINEWIDTH)
		plt.xlabel('# Epochs')
		plt.ylabel('Loss')
		plt.legend()
		pylab.savefig(PATH_FIGURE+'CrossModel_Test_Set_Loss'+'.png',bbox_inches='tight')
		# plt.show()



		#PLOT Per Second
		matplotlib.rcParams.update({'font.size': 16})
		plt.figure(7)
		plt.title('Loss of Validation Set')
		
		if DRAW_MLPBN_ADASGD: 	plt.plot(mlpbn_ADAsgd_epoch_times, mlpbn_ADAsgd_loss_val, SPEC_L1 ,label="AdaGrad",  linewidth = LINEWIDTH)
		if DRAW_MLPBN_streaming: 	plt.plot(mlpbn_streaming_epoch_times, mlpbn_streaming_loss_val, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
		if DRAW_MLPBN_SVRG:	plt.plot(mlpbn_svrg_epoch_times, mlpbn_svrg_loss_val, SPEC_L4 ,label="SVRG",  linewidth = LINEWIDTH)
		plt.xlabel('Seconds')
		plt.ylabel('Loss')
		plt.legend()
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CroszsModel_Validation_Set_Loss'+'.png',bbox_inches='tight')

		plt.figure(8)
		plt.title('Predict Accuracy of Validation Set')
		
		if DRAW_MLPBN_ADASGD: 	plt.plot(mlpbn_ADAsgd_epoch_times, mlpbn_ADAsgd_acc_val, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
		if DRAW_MLPBN_streaming: 	plt.plot(mlpbn_streaming_epoch_times, mlpbn_streaming_acc_val, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
		if DRAW_MLPBN_SVRG:	plt.plot(mlpbn_svrg_epoch_times, mlpbn_svrg_acc_val, SPEC_L4 ,label="SVRG",  linewidth = LINEWIDTH)
		plt.xlabel('Seconds')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CrossModel_Validation_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(9)
		plt.title('Loss of Training Set')
		# if Y_LIM_FINE_TUNING:	pylab.ylim([-0.01,0.25])
		
		if DRAW_MLPBN_ADASGD: 	plt.plot(mlpbn_ADAsgd_epoch_times, mlpbn_ADAsgd_loss_train, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
		if DRAW_MLPBN_streaming: 	plt.plot(mlpbn_streaming_epoch_times, mlpbn_streaming_loss_train, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
		if DRAW_MLPBN_SVRG:	plt.plot(mlpbn_svrg_epoch_times, mlpbn_svrg_loss_train, SPEC_L4 ,label="SVRG", linewidth = LINEWIDTH)
		plt.xlabel('Seconds')
		plt.ylabel('Loss')
		plt.legend()
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CrossModel_Training_Set_Loss'+'.png',bbox_inches='tight')

		plt.figure(10)
		plt.title('Predict Accuracy of Training Set')
		if Y_LIM_FINE_TUNING:	pylab.ylim([0.93,1.01])
		
		if DRAW_MLPBN_ADASGD: 	plt.plot(mlpbn_ADAsgd_epoch_times, mlpbn_ADAsgd_acc_train, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
		if DRAW_MLPBN_streaming: 	plt.plot(mlpbn_streaming_epoch_times, mlpbn_streaming_acc_train, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
		if DRAW_MLPBN_SVRG:	plt.plot(mlpbn_svrg_epoch_times, mlpbn_svrg_acc_train, SPEC_L4 ,label="SVRG",  linewidth = LINEWIDTH)
		plt.xlabel('Seconds')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CrossModel_Training_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(11)
		plt.title('Predict Accuracy of Test Set')
		
		if DRAW_MLPBN_ADASGD: plt.plot(mlpbn_ADAsgd_epoch_times, mlpbn_ADAsgd_acc_test, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
		if DRAW_MLPBN_streaming: 	plt.plot(mlpbn_streaming_epoch_times, mlpbn_streaming_acc_test, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
		if DRAW_MLPBN_SVRG: 	plt.plot(mlpbn_svrg_epoch_times, mlpbn_svrg_acc_test, SPEC_L4 ,label="SVRG", linewidth = LINEWIDTH)
		plt.xlabel('Seconds')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CrossModel_Test_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(12)
		plt.title('Loss of Test Set')
		
		if DRAW_MLPBN_ADASGD: plt.plot(mlpbn_ADAsgd_epoch_times, mlpbn_ADAsgd_loss_test, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
		if DRAW_MLPBN_streaming: 	plt.plot(mlpbn_streaming_epoch_times, mlpbn_streaming_loss_test, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
		if DRAW_MLPBN_SVRG: 	plt.plot(mlpbn_svrg_epoch_times, mlpbn_svrg_loss_test, SPEC_L4 ,label="SVRG", linewidth = LINEWIDTH)
		plt.xlabel('Seconds')
		plt.ylabel('Loss')
		plt.legend()
		pylab.savefig(PATH_FIGURE+'Time_CrossModel_Test_Set_Loss'+'.png',bbox_inches='tight')
		# plt.show()


	if DRAW_BN_PARA:
		bn_para_mu		=	np.loadtxt(PATH_DATA +"log_para_BN_mu.txt")
		bn_para_lambda	=	np.loadtxt(PATH_DATA +"log_para_BN_lambda.txt")

		EPOCH = num_epochs
		NODES = 500

		count_BN_para = np.arange(EPOCH)+1
		bn_para_mu = bn_para_mu.reshape(EPOCH,NODES)
		bn_para_lambda = bn_para_lambda.reshape(EPOCH,NODES)
		# bn_para_mu.reshape(50,500)

		plt.figure(13)
		plt.title('BN Parameters \mu')
		for i in range(NODES):
			plt.plot(count_BN_para, bn_para_mu[:,i], '-',  color=np.random.rand(3,1) , linewidth = 0.1)
		plt.xlabel('Epoch')
		plt.ylabel('Value')
		# plt.legend()
		pylab.savefig(PATH_FIGURE+'BN_Para_MU_Values'+'.png',bbox_inches='tight')

		plt.figure(14)
		plt.title('BN Parameters \lambda')
		for i in range(NODES):
			plt.plot(count_BN_para, bn_para_lambda[:,i], '-',  color=np.random.rand(3,1) , linewidth = 0.1)
		plt.xlabel('Epoch')
		plt.ylabel('Value')
		# plt.legend()
		pylab.savefig(PATH_FIGURE+'BN_Para_LAMBDA_Values'+'.png',bbox_inches='tight')

		
		

	
	print ("Finish drawing cross model plots.")

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv) or ('help' in sys.argv):
        print("arg: NUM_EPOCHS")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        main(**kwargs)


