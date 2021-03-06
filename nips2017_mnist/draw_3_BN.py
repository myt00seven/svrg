# This is used to draw three comparisons for SGD+BN, SVRG+BN and Streaming SVRG +BN

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg') 
import pylab

import numpy as np
		
import sys

# all the methods are with BN layers!!!


PATH_DATA_adagrad  = "data_3_BN/"
PATH_DATA_SVRG     = "data_3_BN/"
PATH_DATA_Stream   = "data_3_BN/"
PATH_DATA   = "data_3_BN/"

PATH_FIGURE = "figure_3/"

MAXLENGTH = 1000
STARTPOINT = 10
LINEWIDTH = 3

LOAD_SGD = True
LOAD_SVRG = True

DRAW_MLPBN_ADASGD = True
DRAW_MLPBN_StreamingSVRG = True
DRAW_MLPBN_SVRG = True

Y_LIM_FINE_TUNING = True

# SPEC_L1 = 'bo-'
# SPEC_L1 = 'g^--'
# SPEC_L3 = 'cs:'
# SPEC_L4 = 'r*-.'

SPEC_L1 = 'b-'
SPEC_L2 = 'g:'
SPEC_L3 = 'r-.'
SPEC_L4 = 'c--'

NUM_EPOCHS = 1000

def main(num_epochs=NUM_EPOCHS):

	str_epochs = str(num_epochs)

	# if LOAD_SGD: mlpbn_ADAsgd_acc_test=			np.loadtxt(PATH_DATA +"_mlpbnTrue_adagrad_acc_test.txt")
	# if LOAD_SGD: mlpbn_ADAsgd_acc_train=		np.loadtxt(PATH_DATA +"_mlpbnTrue_adagrad_acc_train.txt")
	if LOAD_SGD: mlpbn_ADAsgd_acc_val=			np.loadtxt(PATH_DATA +"_mlpbnTrue_adagrad_acc_val.txt")
	# if LOAD_SGD: mlpbn_ADAsgd_loss_test=		np.loadtxt(PATH_DATA +"_mlpbnTrue_adagrad_loss_test.txt")
	if LOAD_SGD: mlpbn_ADAsgd_loss_train=		np.loadtxt(PATH_DATA +"_mlpbnTrue_adagrad_loss_train.txt")
	if LOAD_SGD: mlpbn_ADAsgd_loss_val=			np.loadtxt(PATH_DATA +"_mlpbnTrue_adagrad_loss_val.txt")

	if LOAD_SVRG: mlpbn_Streamingsvrg_acc_train=		np.loadtxt(PATH_DATA_SVRG +"_mlpbnTrue_StreamingSVRG_acc_train.txt")
	if LOAD_SVRG: mlpbn_Streamingsvrg_acc_val=			np.loadtxt(PATH_DATA_SVRG +"_mlpbnTrue_StreamingSVRG_acc_val.txt")
	if LOAD_SVRG: mlpbn_Streamingsvrg_loss_train=		np.loadtxt(PATH_DATA_SVRG +"_mlpbnTrue_StreamingSVRG_loss_train.txt")
	if LOAD_SVRG: mlpbn_Streamingsvrg_loss_val=			np.loadtxt(PATH_DATA_SVRG +"_mlpbnTrue_StreamingSVRG_loss_val.txt")
	if LOAD_SVRG: mlpbn_Streamingsvrg_acc_test=			np.loadtxt(PATH_DATA_SVRG +"_mlpbnTrue_StreamingSVRG_acc_test.txt")
	if LOAD_SVRG: mlpbn_Streamingsvrg_loss_test=		np.loadtxt(PATH_DATA_SVRG +"_mlpbnTrue_StreamingSVRG_loss_test.txt")
	
	if LOAD_SVRG: mlpbn_svrg_acc_train=		np.loadtxt(PATH_DATA +"_mlpbnTrue_SVRG_acc_train.txt")
	if LOAD_SVRG: mlpbn_svrg_acc_val=		np.loadtxt(PATH_DATA +"_mlpbnTrue_SVRG_acc_val.txt")
	if LOAD_SVRG: mlpbn_svrg_loss_train=	np.loadtxt(PATH_DATA +"_mlpbnTrue_SVRG_loss_train.txt")
	if LOAD_SVRG: mlpbn_svrg_loss_val=		np.loadtxt(PATH_DATA +"_mlpbnTrue_SVRG_loss_val.txt")
	if LOAD_SVRG: mlpbn_svrg_acc_test=		np.loadtxt(PATH_DATA +"_mlpbnTrue_SVRG_acc_test.txt")
	if LOAD_SVRG: mlpbn_svrg_loss_test=		np.loadtxt(PATH_DATA +"_mlpbnTrue_SVRG_loss_test.txt")

	count_mlpbn_ADAsgd = 100
	count_mlpbn_Streamingsvrg = 100
	count_mlpbn_svrg = 100

	if DRAW_MLPBN_ADASGD: 	count_mlpbn_ADAsgd = np.arange(mlpbn_ADAsgd_acc_val.shape[0])+1
	if DRAW_MLPBN_StreamingSVRG: 	count_mlpbn_Streamingsvrg = np.arange(mlpbn_Streamingsvrg_acc_train.shape[0])+1
	if DRAW_MLPBN_SVRG: count_mlpbn_svrg = np.arange(mlpbn_svrg_acc_val.shape[0])+1

	# print mlp_sgd_acc_train


	# if (MAXLENGTH>0 or STARTPOINT>0):
		
	# 	if DRAW_MLPBN_ADASGD: 	count_mlpbn_ADAsgd = count_mlpbn_ADAsgd[STARTPOINT:MAXLENGTH+1]
	# 	if DRAW_MLPbn_StreamingSVRG: 	count_mlpbn_Streamingsvrg = count_mlpbn_Streamingsvrg[STARTPOINT:MAXLENGTH+1]
	# 	if DRAW_MLPBN_SVRG: count_mlpbn_svrg = count_mlpbn_svrg[STARTPOINT:MAXLENGTH+1]		

		
	# 	if DRAW_MLPBN_ADASGD: 	mlpbn_ADAsgd_acc_test = mlpbn_ADAsgd_acc_test[STARTPOINT:MAXLENGTH+1]
	# 	if DRAW_MLPbn_StreamingSVRG: 	mlpbn_Streamingsvrg_acc_test = mlpbn_Streamingsvrg_acc_test[STARTPOINT:MAXLENGTH+1]
	# 	if DRAW_MLPBN_SVRG: mlpbn_svrg_acc_test = mlpbn_svrg_acc_test[STARTPOINT:MAXLENGTH+1]		

		
	# 	if DRAW_MLPBN_ADASGD: 	mlpbn_ADAsgd_loss_test = mlpbn_ADAsgd_loss_test[STARTPOINT:MAXLENGTH+1]
	# 	if DRAW_MLPbn_StreamingSVRG: 	mlpbn_Streamingsvrg_loss_test = mlpbn_Streamingsvrg_loss_test[STARTPOINT:MAXLENGTH+1]
	# 	if DRAW_MLPBN_SVRG: mlpbn_svrg_loss_test = mlpbn_svrg_loss_test[STARTPOINT:MAXLENGTH+1]

		
	# 	if DRAW_MLPBN_ADASGD: 	mlpbn_ADAsgd_acc_val = mlpbn_ADAsgd_acc_val[STARTPOINT:MAXLENGTH+1]
	# 	if DRAW_MLPbn_StreamingSVRG: 	mlpbn_Streamingsvrg_acc_val = mlpbn_Streamingsvrg_acc_val[STARTPOINT:MAXLENGTH+1]
	# 	if DRAW_MLPBN_SVRG: mlpbn_svrg_acc_val = mlpbn_svrg_acc_val[STARTPOINT:MAXLENGTH+1]		
		
		
	# 	if DRAW_MLPBN_ADASGD: 	mlpbn_ADAsgd_loss_val = mlpbn_ADAsgd_loss_val[STARTPOINT:MAXLENGTH+1]
	# 	if DRAW_MLPbn_StreamingSVRG: 	mlpbn_Streamingsvrg_loss_val = mlpbn_Streamingsvrg_loss_val[STARTPOINT:MAXLENGTH+1]
	# 	if DRAW_MLPBN_SVRG: mlpbn_svrg_loss_val = mlpbn_svrg_loss_val[STARTPOINT:MAXLENGTH+1]

		
	# 	if DRAW_MLPBN_ADASGD: 	mlpbn_ADAsgd_acc_train = mlpbn_ADAsgd_acc_train[STARTPOINT:MAXLENGTH+1]
	# 	if DRAW_MLPbn_StreamingSVRG: 	mlpbn_Streamingsvrg_acc_train = mlpbn_Streamingsvrg_acc_train[STARTPOINT:MAXLENGTH+1]
	# 	if DRAW_MLPBN_SVRG: mlpbn_svrg_acc_train = mlpbn_svrg_acc_train[STARTPOINT:MAXLENGTH+1]		
		
		
	# 	if DRAW_MLPBN_ADASGD: 	mlpbn_ADAsgd_loss_train = mlpbn_ADAsgd_loss_train[STARTPOINT:MAXLENGTH+1]
	# 	if DRAW_MLPbn_StreamingSVRG: 	mlpbn_Streamingsvrg_loss_train = mlpbn_Streamingsvrg_loss_train[STARTPOINT:MAXLENGTH+1]
	# 	if DRAW_MLPBN_SVRG: mlpbn_svrg_loss_train = mlpbn_svrg_loss_train[STARTPOINT:MAXLENGTH+1]



	#PLOT 
	matplotlib.rcParams.update({'font.size': 16})
	plt.figure(1)
	plt.title('Loss of Validation Set')
	
	if DRAW_MLPBN_ADASGD: 	plt.plot(count_mlpbn_ADAsgd, mlpbn_ADAsgd_loss_val, SPEC_L1 ,label="AdaGrad",  linewidth = LINEWIDTH)
	if DRAW_MLPBN_StreamingSVRG: 	plt.plot(count_mlpbn_Streamingsvrg, mlpbn_Streamingsvrg_loss_val, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
	if DRAW_MLPBN_SVRG:	plt.plot(count_mlpbn_svrg, mlpbn_svrg_loss_val, SPEC_L4 ,label="SVRG",  linewidth = LINEWIDTH)
	plt.xlabel('# Epochs')
	plt.ylabel('Loss')
	plt.legend()
	# plt.show()
	pylab.savefig(PATH_FIGURE+'CroszsModel_Validation_Set_Loss'+'.png',bbox_inches='tight')

	plt.figure(2)
	plt.title('Predict Accuracy of Validation Set')
	
	if DRAW_MLPBN_ADASGD: 	plt.plot(count_mlpbn_ADAsgd, mlpbn_ADAsgd_acc_val, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
	if DRAW_MLPBN_StreamingSVRG: 	plt.plot(count_mlpbn_Streamingsvrg, mlpbn_Streamingsvrg_acc_val, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
	if DRAW_MLPBN_SVRG:	plt.plot(count_mlpbn_svrg, mlpbn_svrg_acc_val, SPEC_L4 ,label="SVRG",  linewidth = LINEWIDTH)
	plt.xlabel('# Epochs')
	plt.ylabel('Predict Accuracy')
	plt.legend(bbox_to_anchor=(1,0.4))
	# plt.show()
	pylab.savefig(PATH_FIGURE+'CrossModel_Validation_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

	plt.figure(3)
	plt.title('Loss of Training Set')
	if Y_LIM_FINE_TUNING:	pylab.ylim([-0.01,0.25])
	
	if DRAW_MLPBN_ADASGD: 	plt.plot(count_mlpbn_ADAsgd, mlpbn_ADAsgd_loss_train, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
	if DRAW_MLPBN_StreamingSVRG: 	plt.plot(count_mlpbn_Streamingsvrg, mlpbn_Streamingsvrg_loss_train, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
	if DRAW_MLPBN_SVRG:	plt.plot(count_mlpbn_svrg, mlpbn_svrg_loss_train, SPEC_L4 ,label="SVRG", linewidth = LINEWIDTH)
	plt.xlabel('# Epochs')
	plt.ylabel('Loss')
	plt.legend()
	# plt.show()
	pylab.savefig(PATH_FIGURE+'CrossModel_Training_Set_Loss'+'.png',bbox_inches='tight')

	# plt.figure(4)
	# plt.title('Predict Accuracy of Training Set')
	# if Y_LIM_FINE_TUNING:	pylab.ylim([0.93,1.01])
	
	# if DRAW_MLPBN_ADASGD: 	plt.plot(count_mlpbn_ADAsgd, mlpbn_ADAsgd_acc_train, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
	# if DRAW_MLPbn_StreamingSVRG: 	plt.plot(count_mlpbn_Streamingsvrg, mlpbn_Streamingsvrg_acc_train, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
	# if DRAW_MLPBN_SVRG:	plt.plot(count_mlpbn_svrg, mlpbn_svrg_acc_train, SPEC_L4 ,label="SVRG",  linewidth = LINEWIDTH)
	# plt.xlabel('# Epochs')
	# plt.ylabel('Predict Accuracy')
	# plt.legend(bbox_to_anchor=(1,0.4))
	# # plt.show()
	# pylab.savefig(PATH_FIGURE+'CrossModel_Training_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

	# plt.figure(5)
	# plt.title('Predict Accuracy of Test Set')
	
	# if DRAW_MLPBN_ADASGD: plt.plot(count_mlpbn_ADAsgd, mlpbn_ADAsgd_acc_test, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
	# if DRAW_MLPbn_StreamingSVRG: 	plt.plot(count_mlpbn_Streamingsvrg, mlpbn_Streamingsvrg_acc_test, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
	# if DRAW_MLPBN_SVRG: 	plt.plot(count_mlpbn_svrg, mlpbn_svrg_acc_test, SPEC_L4 ,label="SVRG", linewidth = LINEWIDTH)
	# plt.xlabel('# Epochs')
	# plt.ylabel('Predict Accuracy')
	# plt.legend(bbox_to_anchor=(1,0.4))
	# # plt.show()
	# pylab.savefig(PATH_FIGURE+'CrossModel_Test_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

	# plt.figure(6)
	# plt.title('Loss of Test Set')
	
	# if DRAW_MLPBN_ADASGD: plt.plot(count_mlpbn_ADAsgd, mlpbn_ADAsgd_loss_test, SPEC_L1 ,label="AdaGrad", linewidth = LINEWIDTH)
	# if DRAW_MLPbn_StreamingSVRG: 	plt.plot(count_mlpbn_Streamingsvrg, mlpbn_Streamingsvrg_loss_test, SPEC_L3 ,label="Streaming SVRG",  linewidth = LINEWIDTH)
	# if DRAW_MLPBN_SVRG: 	plt.plot(count_mlpbn_svrg, mlpbn_svrg_loss_test, SPEC_L4 ,label="SVRG", linewidth = LINEWIDTH)
	# plt.xlabel('# Epochs')
	# plt.ylabel('Loss')
	# plt.legend()
	# pylab.savefig(PATH_FIGURE+'CrossModel_Test_Set_Loss'+'.png',bbox_inches='tight')
	# # plt.show()
	
	print ("Finish drawing cross model plots.")

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv) or ('help' in sys.argv):
        print("arg: NUM_EPOCHS")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        main(**kwargs)


