import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg') 
import pylab

import numpy as np
		
import sys

PATH_DATA  = "data_large/"
PATH_DATA_SVRG  = "../svrg/data/"
PATH_DATA_SVRG_BN  = "../svrg_bn/data/"
PATH_DATA  = "data_large/"

PATH_FIGURE = "figure/"

MAXLENGTH = 1000
STARTPOINT = 10
LINEWIDTH = 3

LOAD_SGD = True
LOAD_SVRG = True

DRAW_MLP_SGD = True
DRAW_MLPBN_SGD = True
DRAW_MLP_SVRG = True
DRAW_MLPBN_SVRG = True

Y_LIM_FINE_TUNING = True

# SPEC_L1 = 'bo-'
# SPEC_L2 = 'g^--'
# SPEC_L3 = 'cs:'
# SPEC_L4 = 'r*-.'

SPEC_L1 = 'b-'
SPEC_L2 = 'g:'
SPEC_L3 = 'r-.'
SPEC_L4 = 'c--'

NUM_EPOCHS = 1000

def main(num_epochs=NUM_EPOCHS):

	str_epochs = str(num_epochs)

	if LOAD_SGD: mlp_sgd_loss_train=		np.loadtxt(PATH_DATA +"mlp_sgd_"+str_epochs+"_loss_train.txt")
	if LOAD_SGD: mlp_sgd_loss_val=			np.loadtxt(PATH_DATA +"mlp_sgd_"+str_epochs+"_loss_val.txt")
	if LOAD_SGD: mlp_sgd_loss_test=			np.loadtxt(PATH_DATA +"mlp_sgd_"+str_epochs+"_loss_test.txt")
	if LOAD_SGD: mlp_sgd_acc_train=			np.loadtxt(PATH_DATA +"mlp_sgd_"+str_epochs+"_acc_train.txt")
	if LOAD_SGD: mlp_sgd_acc_val=			np.loadtxt(PATH_DATA +"mlp_sgd_"+str_epochs+"_acc_val.txt")
	if LOAD_SGD: mlp_sgd_acc_test=			np.loadtxt(PATH_DATA +"mlp_sgd_"+str_epochs+"_acc_test.txt")

	if LOAD_SGD: mlpbn_sgd_acc_test=		np.loadtxt(PATH_DATA +"mlpbn_sgd_"+str_epochs+"_acc_test.txt")
	if LOAD_SGD: mlpbn_sgd_acc_train=		np.loadtxt(PATH_DATA +"mlpbn_sgd_"+str_epochs+"_acc_train.txt")
	if LOAD_SGD: mlpbn_sgd_acc_val=			np.loadtxt(PATH_DATA +"mlpbn_sgd_"+str_epochs+"_acc_val.txt")
	if LOAD_SGD: mlpbn_sgd_loss_test=		np.loadtxt(PATH_DATA +"mlpbn_sgd_"+str_epochs+"_loss_test.txt")
	if LOAD_SGD: mlpbn_sgd_loss_train=		np.loadtxt(PATH_DATA +"mlpbn_sgd_"+str_epochs+"_loss_train.txt")
	if LOAD_SGD: mlpbn_sgd_loss_val=		np.loadtxt(PATH_DATA +"mlpbn_sgd_"+str_epochs+"_loss_val.txt")

	if LOAD_SVRG: mlp_svrg_acc_train=		np.loadtxt(PATH_DATA_SVRG +"_mlpbnFalse_SVRG_acc_train.txt")
	if LOAD_SVRG: mlp_svrg_acc_val=			np.loadtxt(PATH_DATA_SVRG +"_mlpbnFalse_SVRG_acc_val.txt")
	if LOAD_SVRG: mlp_svrg_loss_train=		np.loadtxt(PATH_DATA_SVRG +"_mlpbnFalse_SVRG_loss_train.txt")
	if LOAD_SVRG: mlp_svrg_loss_val=		np.loadtxt(PATH_DATA_SVRG +"_mlpbnFalse_SVRG_loss_val.txt")
	if LOAD_SVRG: mlp_svrg_acc_test=		np.loadtxt(PATH_DATA_SVRG +"_mlpbnFalse_SVRG_acc_test.txt")
	if LOAD_SVRG: mlp_svrg_loss_test=		np.loadtxt(PATH_DATA_SVRG +"_mlpbnFalse_SVRG_loss_test.txt")
	
	if LOAD_SVRG: mlpbn_svrg_acc_train=		np.loadtxt(PATH_DATA_SVRG_BN +"_mlpbnTrue_SVRG_acc_train.txt")
	if LOAD_SVRG: mlpbn_svrg_acc_val=		np.loadtxt(PATH_DATA_SVRG_BN +"_mlpbnTrue_SVRG_acc_val.txt")
	if LOAD_SVRG: mlpbn_svrg_loss_train=	np.loadtxt(PATH_DATA_SVRG_BN +"_mlpbnTrue_SVRG_loss_train.txt")
	if LOAD_SVRG: mlpbn_svrg_loss_val=		np.loadtxt(PATH_DATA_SVRG_BN +"_mlpbnTrue_SVRG_loss_val.txt")
	if LOAD_SVRG: mlpbn_svrg_acc_test=		np.loadtxt(PATH_DATA_SVRG_BN +"_mlpbnTrue_SVRG_acc_test.txt")
	if LOAD_SVRG: mlpbn_svrg_loss_test=		np.loadtxt(PATH_DATA_SVRG_BN +"_mlpbnTrue_SVRG_loss_test.txt")

	if DRAW_MLP_SGD: 	count_mlp_sgd = np.arange(mlp_sgd_acc_train.shape[0])+1
	if DRAW_MLPBN_SGD: 	count_mlpbn_sgd = np.arange(mlpbn_sgd_acc_train.shape[0])+1
	if DRAW_MLP_SVRG: 	count_mlp_svrg = np.arange(mlp_svrg_acc_train.shape[0])+1
	if DRAW_MLPBN_SVRG: count_mlpbn_svrg = np.arange(mlpbn_svrg_acc_train.shape[0])+1

	# print mlp_sgd_acc_train


	if (MAXLENGTH>0 or STARTPOINT>0):
		if DRAW_MLP_SGD: 	count_mlp_sgd = count_mlp_sgd[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLPBN_SGD: 	count_mlpbn_sgd = count_mlpbn_sgd[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLP_SVRG: 	count_mlp_svrg = count_mlp_svrg[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLPBN_SVRG: count_mlpbn_svrg = count_mlpbn_svrg[STARTPOINT:MAXLENGTH+1]		

		if DRAW_MLP_SGD: 	mlp_sgd_acc_test = mlp_sgd_acc_test[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLPBN_SGD: 	mlpbn_sgd_acc_test = mlpbn_sgd_acc_test[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLP_SVRG: 	mlp_svrg_acc_test = mlp_svrg_acc_test[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLPBN_SVRG: mlpbn_svrg_acc_test = mlpbn_svrg_acc_test[STARTPOINT:MAXLENGTH+1]		

		if DRAW_MLP_SGD:	mlp_sgd_loss_test = mlp_sgd_loss_test[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLPBN_SGD: 	mlpbn_sgd_loss_test = mlpbn_sgd_loss_test[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLP_SVRG: 	mlp_svrg_loss_test = mlp_svrg_loss_test[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLPBN_SVRG: mlpbn_svrg_loss_test = mlpbn_svrg_loss_test[STARTPOINT:MAXLENGTH+1]

		if DRAW_MLP_SGD: 	mlp_sgd_acc_val = mlp_sgd_acc_val[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLPBN_SGD: 	mlpbn_sgd_acc_val = mlpbn_sgd_acc_val[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLP_SVRG: 	mlp_svrg_acc_val = mlp_svrg_acc_val[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLPBN_SVRG: mlpbn_svrg_acc_val = mlpbn_svrg_acc_val[STARTPOINT:MAXLENGTH+1]		
		
		if DRAW_MLP_SGD:	mlp_sgd_loss_val = mlp_sgd_loss_val[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLPBN_SGD: 	mlpbn_sgd_loss_val = mlpbn_sgd_loss_val[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLP_SVRG: 	mlp_svrg_loss_val = mlp_svrg_loss_val[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLPBN_SVRG: mlpbn_svrg_loss_val = mlpbn_svrg_loss_val[STARTPOINT:MAXLENGTH+1]

		if DRAW_MLP_SGD: 	mlp_sgd_acc_train = mlp_sgd_acc_train[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLPBN_SGD: 	mlpbn_sgd_acc_train = mlpbn_sgd_acc_train[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLP_SVRG: 	mlp_svrg_acc_train = mlp_svrg_acc_train[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLPBN_SVRG: mlpbn_svrg_acc_train = mlpbn_svrg_acc_train[STARTPOINT:MAXLENGTH+1]		
		
		if DRAW_MLP_SGD:	mlp_sgd_loss_train = mlp_sgd_loss_train[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLPBN_SGD: 	mlpbn_sgd_loss_train = mlpbn_sgd_loss_train[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLP_SVRG: 	mlp_svrg_loss_train = mlp_svrg_loss_train[STARTPOINT:MAXLENGTH+1]
		if DRAW_MLPBN_SVRG: mlpbn_svrg_loss_train = mlpbn_svrg_loss_train[STARTPOINT:MAXLENGTH+1]



	#PLOT 
	matplotlib.rcParams.update({'font.size': 16})
	plt.figure(1)
	plt.title('Loss of Validation Set')
	if DRAW_MLP_SGD: 	plt.plot(count_mlp_sgd, mlp_sgd_loss_val, SPEC_L1 ,label="MLP SGD",  linewidth = LINEWIDTH)
	if DRAW_MLPBN_SGD: 	plt.plot(count_mlpbn_sgd, mlpbn_sgd_loss_val, SPEC_L2 ,label="MLPBN SGD",  linewidth = LINEWIDTH)
	if DRAW_MLP_SVRG: 	plt.plot(count_mlp_svrg, mlp_svrg_loss_val, SPEC_L3 ,label="MLP SVRG",  linewidth = LINEWIDTH)
	if DRAW_MLPBN_SVRG:	plt.plot(count_mlpbn_svrg, mlpbn_svrg_loss_val, SPEC_L4 ,label="MLPBN SVRG",  linewidth = LINEWIDTH)
	plt.xlabel('# Epochs')
	plt.ylabel('Loss')
	plt.legend()
	# plt.show()
	pylab.savefig(PATH_FIGURE+'CroszsModel_Validation_Set_Loss'+'.png',bbox_inches='tight')

	plt.figure(2)
	plt.title('Predict Accuracy of Validation Set')
	if DRAW_MLP_SGD: 	plt.plot(count_mlp_sgd, mlp_sgd_acc_val, SPEC_L1 ,label="MLP SGD", linewidth = LINEWIDTH)
	if DRAW_MLPBN_SGD: 	plt.plot(count_mlpbn_sgd, mlpbn_sgd_acc_val, SPEC_L2 ,label="MLPBN SGD", linewidth = LINEWIDTH)
	if DRAW_MLP_SVRG: 	plt.plot(count_mlp_svrg, mlp_svrg_acc_val, SPEC_L3 ,label="MLP SVRG",  linewidth = LINEWIDTH)
	if DRAW_MLPBN_SVRG:	plt.plot(count_mlpbn_svrg, mlpbn_svrg_acc_val, SPEC_L4 ,label="MLPBN SVRG",  linewidth = LINEWIDTH)
	plt.xlabel('# Epochs')
	plt.ylabel('Predict Accuracy')
	plt.legend(bbox_to_anchor=(1,0.4))
	# plt.show()
	pylab.savefig(PATH_FIGURE+'CrossModel_Validation_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

	plt.figure(3)
	plt.title('Loss of Training Set')
	if Y_LIM_FINE_TUNING:	pylab.ylim([-0.01,0.25])
	if DRAW_MLP_SGD: 	plt.plot(count_mlp_sgd, mlp_sgd_loss_train, SPEC_L1 ,label="MLP SGD", linewidth = LINEWIDTH)
	if DRAW_MLPBN_SGD: 	plt.plot(count_mlpbn_sgd, mlpbn_sgd_loss_train, SPEC_L2 ,label="MLPBN SGD", linewidth = LINEWIDTH)
	if DRAW_MLP_SVRG: 	plt.plot(count_mlp_svrg, mlp_svrg_loss_train, SPEC_L3 ,label="MLP SVRG",  linewidth = LINEWIDTH)
	if DRAW_MLPBN_SVRG:	plt.plot(count_mlpbn_svrg, mlpbn_svrg_loss_train, SPEC_L4 ,label="MLPBN SVRG", linewidth = LINEWIDTH)
	plt.xlabel('# Epochs')
	plt.ylabel('Loss')
	plt.legend()
	# plt.show()
	pylab.savefig(PATH_FIGURE+'CrossModel_Training_Set_Loss'+'.png',bbox_inches='tight')

	plt.figure(4)
	plt.title('Predict Accuracy of Training Set')
	if Y_LIM_FINE_TUNING:	pylab.ylim([0.93,1.01])
	if DRAW_MLP_SGD: 	plt.plot(count_mlp_sgd, mlp_sgd_acc_train, SPEC_L1 ,label="MLP SGD", linewidth = LINEWIDTH)
	if DRAW_MLPBN_SGD: 	plt.plot(count_mlpbn_sgd, mlpbn_sgd_acc_train, SPEC_L2 ,label="MLPBN SGD", linewidth = LINEWIDTH)
	if DRAW_MLP_SVRG: 	plt.plot(count_mlp_svrg, mlp_svrg_acc_train, SPEC_L3 ,label="MLP SVRG",  linewidth = LINEWIDTH)
	if DRAW_MLPBN_SVRG:	plt.plot(count_mlpbn_svrg, mlpbn_svrg_acc_train, SPEC_L4 ,label="MLPBN SVRG",  linewidth = LINEWIDTH)
	plt.xlabel('# Epochs')
	plt.ylabel('Predict Accuracy')
	plt.legend(bbox_to_anchor=(1,0.4))
	# plt.show()
	pylab.savefig(PATH_FIGURE+'CrossModel_Training_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

	plt.figure(5)
	plt.title('Predict Accuracy of Test Set')
	if DRAW_MLP_SGD: plt.plot(count_mlp_sgd, mlp_sgd_acc_test, SPEC_L1 ,label="MLP SGD", linewidth = LINEWIDTH)
	if DRAW_MLPBN_SGD: plt.plot(count_mlpbn_sgd, mlpbn_sgd_acc_test, SPEC_L2 ,label="MLPBN SGD", linewidth = LINEWIDTH)
	if DRAW_MLP_SVRG: 	plt.plot(count_mlp_svrg, mlp_svrg_acc_test, SPEC_L3 ,label="MLP SVRG",  linewidth = LINEWIDTH)
	if DRAW_MLPBN_SVRG: 	plt.plot(count_mlpbn_svrg, mlpbn_svrg_acc_test, SPEC_L4 ,label="MLPBN SVRG", linewidth = LINEWIDTH)
	plt.xlabel('# Epochs')
	plt.ylabel('Predict Accuracy')
	plt.legend(bbox_to_anchor=(1,0.4))
	# plt.show()
	pylab.savefig(PATH_FIGURE+'CrossModel_Test_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

	plt.figure(6)
	plt.title('Loss of Test Set')
	if DRAW_MLP_SGD: plt.plot(count_mlp_sgd, mlp_sgd_loss_test, SPEC_L1 ,label="MLP SGD", linewidth = LINEWIDTH)
	if DRAW_MLPBN_SGD: plt.plot(count_mlpbn_sgd, mlpbn_sgd_loss_test, SPEC_L2 ,label="MLPBN SGD", linewidth = LINEWIDTH)
	if DRAW_MLP_SVRG: 	plt.plot(count_mlp_svrg, mlp_svrg_loss_test, SPEC_L3 ,label="MLP SVRG",  linewidth = LINEWIDTH)
	if DRAW_MLPBN_SVRG: 	plt.plot(count_mlpbn_svrg, mlpbn_svrg_loss_test, SPEC_L4 ,label="MLPBN SVRG", linewidth = LINEWIDTH)
	plt.xlabel('# Epochs')
	plt.ylabel('Loss')
	plt.legend()
	pylab.savefig(PATH_FIGURE+'CrossModel_Test_Set_Loss'+'.png',bbox_inches='tight')
	# plt.show()
	
	print ("Finish drawing cross model plots.")

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv) or ('help' in sys.argv):
        print("arg: NUM_EPOCHS")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        main(**kwargs)


