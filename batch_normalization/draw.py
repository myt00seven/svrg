import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg') 
import pylab

import numpy as np

import sys

PATH_DATA  = "data/"
PATH_DATA_SVRG  = "data_svrg/"
PATH_DATA_LARGE_SCALE  = "data_large/"
PATH_FIGURE = "figure/"

DRAW_MLP_SGD = True
DRAW_MLPBN_SGD = True
DRAW_MLP_SVRG = False
DRAW_MLPBN_SVRG = False

NUM_EPOCHS = 1000

def main(num_epochs=NUM_EPOCHS):

	str_epochs = str(num_epochs)

	mlp_sgd_acc_train=		np.loadtxt( PATH_DATA +"mlp_sgd_acc_train.txt")
	mlp_sgd_acc_val=		np.loadtxt( PATH_DATA +"mlp_sgd_acc_val.txt")
	mlp_sgd_loss_train=		np.loadtxt( PATH_DATA +"mlp_sgd_loss_train.txt")
	mlp_sgd_loss_val=		np.loadtxt( PATH_DATA +"mlp_sgd_loss_val.txt")
	mlpbn_sgd_acc_train=	np.loadtxt( PATH_DATA +"mlpbn_sgd_acc_train.txt")
	mlpbn_sgd_acc_val=		np.loadtxt( PATH_DATA +"mlpbn_sgd_acc_val.txt")
	mlpbn_sgd_loss_train=	np.loadtxt( PATH_DATA +"mlpbn_sgd_loss_train.txt")
	mlpbn_sgd_loss_val=		np.loadtxt( PATH_DATA +"mlpbn_sgd_loss_val.txt")
	mlp_svrg_acc_train=		np.loadtxt( PATH_DATA_SVRG +"_mlpbnFalse_SVRG_acc_train.txt")
	mlp_svrg_acc_val=		np.loadtxt( PATH_DATA_SVRG +"_mlpbnFalse_SVRG_acc_val.txt")
	mlp_svrg_loss_train=	np.loadtxt( PATH_DATA_SVRG +"_mlpbnFalse_SVRG_loss_train.txt")
	mlp_svrg_loss_val=		np.loadtxt( PATH_DATA_SVRG +"_mlpbnFalse_SVRG_loss_val.txt")
	mlpbn_svrg_acc_train=	np.loadtxt( PATH_DATA_SVRG +"_mlpbnTrue_SVRG_acc_train.txt")
	mlpbn_svrg_acc_val=		np.loadtxt( PATH_DATA_SVRG +"_mlpbnTrue_SVRG_acc_val.txt")
	mlpbn_svrg_loss_train=	np.loadtxt( PATH_DATA_SVRG +"_mlpbnTrue_SVRG_loss_train.txt")
	mlpbn_svrg_loss_val=	np.loadtxt( PATH_DATA_SVRG +"_mlpbnTrue_SVRG_loss_val.txt")

	mlp_sgd_acc_test=		np.loadtxt( PATH_DATA_LARGE_SCALE +"mlp_sgd_"+str_epochs+"_acc_test.txt")
	mlp_sgd_loss_test=		np.loadtxt( PATH_DATA_LARGE_SCALE +"mlp_sgd_"+str_epochs+"_loss_test.txt")
	mlpbn_sgd_acc_test=		np.loadtxt( PATH_DATA_LARGE_SCALE +"mlpbn_sgd_"+str_epochs+"_acc_test.txt")
	mlpbn_sgd_loss_test=	np.loadtxt( PATH_DATA_LARGE_SCALE +"mlpbn_sgd_"+str_epochs+"_loss_test.txt")

	count = np.arange(mlp_sgd_acc_train.shape[0])+1

	# print mlp_sgd_acc_train

	#PLOT 
	matplotlib.rcParams.update({'font.size': 16})
	plt.figure(1)
	plt.title('Loss of Validation Set')
	if DRAW_MLP_SGD: 	plt.plot(count, mlp_sgd_loss_val, 'bo-',label="MLP SGD", linewidth='1')
	if DRAW_MLPBN_SGD: 	plt.plot(count, mlpbn_sgd_loss_val, 'g^--',label="MLPBN SGD", linewidth='1')
	if DRAW_MLP_SVRG: 	plt.plot(count, mlp_svrg_loss_val, 'cs:',label="MLP SVRG", linewidth='1')
	if DRAW_MLPBN_SVRG:	plt.plot(count, mlpbn_svrg_loss_val, 'r*-.',label="MLPBN SVRG", linewidth='1')
	plt.xlabel('# Epochs')
	plt.ylabel('Loss')
	plt.legend()
	# plt.show()
	pylab.savefig(PATH_FIGURE+'CrossModel_Validation Set_Loss'+'.png',bbox_inches='tight')

	plt.figure(2)
	plt.title('Predict Accuracy of Validation Set')
	if DRAW_MLP_SGD: 	plt.plot(count, mlp_sgd_acc_val, 'bo-',label="MLP SGD")
	if DRAW_MLPBN_SGD: 	plt.plot(count, mlpbn_sgd_acc_val, 'g^--',label="MLPBN SGD")
	if DRAW_MLP_SVRG: 	plt.plot(count, mlp_svrg_acc_val, 'cs:',label="MLP SVRG", linewidth='1')
	if DRAW_MLPBN_SVRG:	plt.plot(count, mlpbn_svrg_acc_val, 'r*-.',label="MLPBN SVRG", linewidth='1')
	plt.xlabel('# Epochs')
	plt.ylabel('Predict Accuracy')
	plt.legend(bbox_to_anchor=(1,0.25))
	# plt.show()
	pylab.savefig(PATH_FIGURE+'CrossModel_Validation Set_Predict Accuracy'+'.png',bbox_inches='tight')

	plt.figure(3)
	plt.title('Loss of Training Set')
	if DRAW_MLP_SGD: 	plt.plot(count, mlp_sgd_loss_train, 'bo-',label="MLP SGD")
	if DRAW_MLPBN_SGD: 	plt.plot(count, mlpbn_sgd_loss_train, 'g^--',label="MLPBN SGD")
	if DRAW_MLP_SVRG: 	plt.plot(count, mlp_svrg_loss_train, 'cs:',label="MLP SVRG", linewidth='1')
	if DRAW_MLPBN_SVRG:	plt.plot(count, mlpbn_svrg_loss_train, 'r*-.',label="MLPBN SVRG", linewidth='1')
	plt.xlabel('# Epochs')
	plt.ylabel('Loss')
	plt.legend()
	# plt.show()
	pylab.savefig(PATH_FIGURE+'CrossModel_Training Set_Loss'+'.png',bbox_inches='tight')

	plt.figure(4)
	plt.title('Predict Accuracy of Training Set')
	if DRAW_MLP_SGD: 	plt.plot(count, mlp_sgd_acc_train, 'bo-',label="MLP SGD")
	if DRAW_MLPBN_SGD: 	plt.plot(count, mlpbn_sgd_acc_train, 'g^--',label="MLPBN SGD")
	if DRAW_MLP_SVRG: 	plt.plot(count, mlp_svrg_acc_train, 'cs:',label="MLP SVRG", linewidth='1')
	if DRAW_MLPBN_SVRG:	plt.plot(count, mlpbn_svrg_acc_train, 'r*-.',label="MLPBN SVRG", linewidth='1')
	plt.xlabel('# Epochs')
	plt.ylabel('Predict Accuracy')
	plt.legend(bbox_to_anchor=(1,0.25))
	# plt.show()
	pylab.savefig(PATH_FIGURE+'CrossModel_Training Set_Predict Accuracy'+'.png',bbox_inches='tight')


	count = np.arange(mlp_sgd_acc_test.shape[0])+1

	plt.figure(5)
	plt.title('Predict Accuracy of Test Set')
	if DRAW_MLP_SGD: plt.plot(count, mlp_sgd_acc_test, 'b-',label="MLP SGD")
	if DRAW_MLPBN_SGD: plt.plot(count, mlpbn_sgd_acc_test, 'g-',label="MLPBN SGD")
	plt.xlabel('# Epochs')
	plt.ylabel('Predict Accuracy')
	plt.legend(bbox_to_anchor=(1,0.25))
	# plt.show()
	pylab.savefig(PATH_FIGURE+'CrossModel_Test Set_Predict Accuracy'+'.png',bbox_inches='tight')

	plt.figure(6)
	plt.title('Loss of Test Set')
	if DRAW_MLP_SGD: plt.plot(count, mlp_sgd_loss_test, 'b-',label="MLP SGD")
	if DRAW_MLPBN_SGD: plt.plot(count, mlpbn_sgd_loss_test, 'g-',label="MLPBN SGD")
	plt.xlabel('# Epochs')
	plt.ylabel('Loss')
	plt.legend()
	pylab.savefig(PATH_FIGURE+'CrossModel_Test Set_Loss'+'.png',bbox_inches='tight')
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


