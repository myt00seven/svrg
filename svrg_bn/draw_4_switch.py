# This is used to draw three comparisons for SGD+BN, SVRG+BN and Streaming SVRG +BN

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg') 
import pylab
import numpy as np
		
import sys

# all the methods are with BN layers!!!

def deminish(a) :
    end_factor=0.4
    length = len(a)
    for i in range(length):
        a[i] = a[i] * (1-(1-end_factor)*(i/float(length)))
    return a

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

PATH_DATA_adagrad  = "data/"
PATH_DATA_SVRG     = "data/"
PATH_DATA_Stream   = "data/"
PATH_DATA   = "data/"

PATH_FIGURE = "figure_3/"

MAXLENGTH = 800
STARTPOINT = 9
LINEWIDTH = 3

DRAW_COMPARE = False

LOAD_SGD = True
LOAD_SVRG = True

DRAW_Line1 = True
DRAW_Line2 = True
DRAW_Line3 = True
DRAW_Line4 = True

DRAW_BN_PARA = True

Y_LIM_FINE_TUNING = True

N_MVA = 5
# Number of Moving Average

# SPEC_L1 = 'bo-'
# SPEC_L1 = 'g^--'
# SPEC_L2 = 'cs:'
# SPEC_L4 = 'r*-.'

SPEC_L1 = 'b-'
SPEC_L2 = 'c:'
SPEC_L3 = 'r-.'
SPEC_L4 = 'g--'

NUM_EPOCHS = 1000

def main(num_epochs=NUM_EPOCHS):

	if DRAW_COMPARE:

		str_epochs = str(num_epochs)

		if LOAD_SGD: Line0.2_acc_test=			np.loadtxt(PATH_DATA +"ratio_0.2_acc_test.txt")
		if LOAD_SGD: Line0.2_acc_train=		np.loadtxt(PATH_DATA +"ratio_0.2_acc_train.txt")
		if LOAD_SGD: Line0.2_acc_val=			np.loadtxt(PATH_DATA +"ratio_0.2_acc_val.txt")
		if LOAD_SGD: Line0.2_loss_test=		np.loadtxt(PATH_DATA +"ratio_0.2_loss_test.txt")
		if LOAD_SGD: Line0.2_loss_train=		np.loadtxt(PATH_DATA +"ratio_0.2_loss_train.txt")
		if LOAD_SGD: Line0.2_loss_val=			np.loadtxt(PATH_DATA +"ratio_0.2_loss_val.txt")
		if LOAD_SGD: Line0.2_epoch_times=			np.loadtxt(PATH_DATA +"ratio_0.2_epoch_times.txt")

		if LOAD_SVRG: Line2_acc_train=		np.loadtxt(PATH_DATA_SVRG +"ratio_0.4_acc_train.txt")
		if LOAD_SVRG: Line2_acc_val=			np.loadtxt(PATH_DATA_SVRG +"ratio_0.4_acc_val.txt")
		if LOAD_SVRG: Line2_loss_train=		np.loadtxt(PATH_DATA_SVRG +"ratio_0.4_loss_train.txt")
		if LOAD_SVRG: Line2_loss_val=			np.loadtxt(PATH_DATA_SVRG +"ratio_0.4_loss_val.txt")
		if LOAD_SVRG: Line2_acc_test=			np.loadtxt(PATH_DATA_SVRG +"ratio_0.4_acc_test.txt")
		if LOAD_SVRG: Line2_loss_test=		np.loadtxt(PATH_DATA_SVRG +"ratio_0.4_loss_test.txt")
		if LOAD_SVRG: Line2_epoch_times=		np.loadtxt(PATH_DATA_SVRG +"ratio_0.4_epoch_times.txt")
		
		if LOAD_SVRG: Line3_acc_train=		np.loadtxt(PATH_DATA +"ratio_0.6_acc_train.txt")
		if LOAD_SVRG: Line3_acc_val=		np.loadtxt(PATH_DATA +"ratio_0.6_acc_val.txt")
		if LOAD_SVRG: Line3_loss_train=		np.loadtxt(PATH_DATA +"ratio_0.6_loss_train.txt")
		if LOAD_SVRG: Line3_loss_val=		np.loadtxt(PATH_DATA +"ratio_0.6_loss_val.txt")
		if LOAD_SVRG: Line3_acc_test=		np.loadtxt(PATH_DATA +"ratio_0.6_acc_test.txt")
		if LOAD_SVRG: Line3_loss_test=		np.loadtxt(PATH_DATA +"ratio_0.6_loss_test.txt")
		if LOAD_SVRG: Line3_epoch_times=		np.loadtxt(PATH_DATA +"ratio_0.6_epoch_times.txt")

		if LOAD_SVRG: Line4_acc_train=		np.loadtxt(PATH_DATA +"ratio_0.8_acc_train.txt")
		if LOAD_SVRG: Line4_acc_val=		np.loadtxt(PATH_DATA +"ratio_0.8_acc_val.txt")
		if LOAD_SVRG: Line4_loss_train=		np.loadtxt(PATH_DATA +"ratio_0.8_loss_train.txt")
		if LOAD_SVRG: Line4_loss_val=		np.loadtxt(PATH_DATA +"ratio_0.8_loss_val.txt")
		if LOAD_SVRG: Line4_acc_test=		np.loadtxt(PATH_DATA +"ratio_0.8_acc_test.txt")
		if LOAD_SVRG: Line4_loss_test=		np.loadtxt(PATH_DATA +"ratio_0.8_loss_test.txt")
		if LOAD_SVRG: Line4_epoch_times=		np.loadtxt(PATH_DATA +"ratio_0.8_epoch_times.txt")

		# count_Line1 = 200
		# count_Line2 = 200
		# count_Line3 = 200

		if DRAW_Line1: 	count_Line1 = np.arange(Line1_acc_val.shape[0])+1
		if DRAW_Line2: 	count_Line2 = np.arange(Line2_acc_train.shape[0])+1
		if DRAW_Line3: count_Line3 = np.arange(Line3_acc_val.shape[0])+1
		if DRAW_Line4: count_Line4 = np.a4ange(Line4_acc_val.shape[0])+1


		# print mlp_sgd_acc_train

		MAXLENGTH = num_epochs
		if (MAXLENGTH>0 or STARTPOINT>0):		# Need add for epoch_times
			if DRAW_Line1: 	count_Line1 = count_Line1[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line2: 	count_Line2 = count_Line2[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line3: count_Line3 = count_Line3[STARTPOINT:MAXLENGTH+1]		
			if DRAW_Line4: count_Line4 = count_Line4[STARTPOINT:MAXLENGTH+1]		

			
			if DRAW_Line1: 	Line1_acc_test = Line1_acc_test[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line2: 	Line2_acc_test = Line2_acc_test[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line3: Line3_acc_test = Line3_acc_test[STARTPOINT:MAXLENGTH+1]		
			if DRAW_Line4: Line4_acc_test = Line4_acc_test[STARTPOINT:MAXLENGTH+1]		

			
			if DRAW_Line1: 	Line1_loss_test = Line1_loss_test[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line2: 	Line2_loss_test = Line2_loss_test[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line3: Line3_loss_test = Line3_loss_test[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line4: Line4_loss_test = Line4_loss_test[STARTPOINT:MAXLENGTH+1]

			
			if DRAW_Line1: 	Line1_acc_val = Line1_acc_val[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line2: 	Line2_acc_val = Line2_acc_val[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line3: Line3_acc_val = Line3_acc_val[STARTPOINT:MAXLENGTH+1]		
			if DRAW_Line4: Line4_acc_val = Line4_acc_val[STARTPOINT:MAXLENGTH+1]		

			
			if DRAW_Line1: 	Line1_loss_val = Line1_loss_val[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line2: 	Line2_loss_val = Line2_loss_val[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line3: Line3_loss_val = Line3_loss_val[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line4: Line4_loss_val = Line4_loss_val[STARTPOINT:MAXLENGTH+1]

			
			if DRAW_Line1: 	Line1_acc_train = Line1_acc_train[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line2: 	Line2_acc_train = Line2_acc_train[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line3: Line3_acc_train = Line3_acc_train[STARTPOINT:MAXLENGTH+1]		
			if DRAW_Line4: Line4_acc_train = Line4_acc_train[STARTPOINT:MAXLENGTH+1]		

					
			if DRAW_Line1: 	Line1_loss_train = Line1_loss_train[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line2: 	Line2_loss_train = Line2_loss_train[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line3: Line3_loss_train = Line3_loss_train[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line4: Line4_loss_train = Line4_loss_train[STARTPOINT:MAXLENGTH+1]


			if DRAW_Line1: 	Line1_epoch_times = Line1_epoch_times[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line2: 	Line2_epoch_times = Line2_epoch_times[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line3: Line3_epoch_times = Line3_epoch_times[STARTPOINT:MAXLENGTH+1]
			if DRAW_Line4: Line4_epoch_times = Line4_epoch_times[STARTPOINT:MAXLENGTH+1]




		#PLOT 
		matplotlib.rcParams.update({'font.size': 16})
		plt.figure(1)
		plt.title('Loss of Validation Set')
		
		if DRAW_Line1: 	plt.plot(count_Line1, Line1_loss_val, SPEC_L1 ,label="Switch 20%",  linewidth = LINEWIDTH)
		if DRAW_Line2: 	plt.plot(count_Line2, Line2_loss_val, SPEC_L2 ,label="Switch 40%",  linewidth = LINEWIDTH)
		if DRAW_Line3:	plt.plot(count_Line3, Line3_loss_val, SPEC_L3 ,label="Switch 60%",  linewidth = LINEWIDTH)
		if DRAW_Line4:	plt.plot(count_Line4, Line4_loss_val, SPEC_L4 ,label="Switch 80%",  linewidth = LINEWIDTH)

		plt.xlabel('# Epochs')
		plt.ylabel('Loss')
		plt.legend()
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CroszsModel_Validation_Set_Loss'+'.png',bbox_inches='tight')

		plt.figure(2)
		plt.title('Predict Accuracy of Validation Set')
		
		if DRAW_Line1: 	plt.plot(count_Line1, Line1_acc_val, SPEC_L1 ,label="Switch 20%", linewidth = LINEWIDTH)
		if DRAW_Line2: 	plt.plot(count_Line2, Line2_acc_val, SPEC_L2 ,label="Switch 40%",  linewidth = LINEWIDTH)
		if DRAW_Line3:	plt.plot(count_Line3, Line3_acc_val, SPEC_L3 ,label="Switch 60%",  linewidth = LINEWIDTH)
		if DRAW_Line4:	plt.plot(count_Line4, Line4_acc_val, SPEC_L4 ,label="Switch 80%",  linewidth = LINEWIDTH)

		plt.xlabel('# Epochs')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CrossModel_Validation_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(3)
		plt.title('Loss of Training Set')
		# if Y_LIM_FINE_TUNING:	pylab.ylim([-0.01,0.25])	
		if DRAW_Line1: 	plt.plot(count_Line1, Line1_loss_train, SPEC_L1 ,label="Switch 20%", linewidth = LINEWIDTH)
		if DRAW_Line2: 	plt.plot(count_Line2, Line2_loss_train, SPEC_L2 ,label="Switch 40%",  linewidth = LINEWIDTH)
		if DRAW_Line3:	plt.plot(count_Line3, Line3_loss_train, SPEC_L3 ,label="Switch 60%", linewidth = LINEWIDTH)
		if DRAW_Line4:	plt.plot(count_Line4, Line4_loss_train, SPEC_L4 ,label="Switch 80%", linewidth = LINEWIDTH)

		plt.xlabel('# Epochs')
		plt.ylabel('Loss')
		plt.legend()
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CrossModel_Training_Set_Loss'+'.png',bbox_inches='tight')

		plt.figure(4)
		plt.title('Predict Accuracy of Training Set')
		if Y_LIM_FINE_TUNING:	pylab.ylim([0.93,1.01])	
		if DRAW_Line1: 	plt.plot(count_Line1, Line1_acc_train, SPEC_L1 ,label="Switch 20%", linewidth = LINEWIDTH)
		if DRAW_Line2: 	plt.plot(count_Line2, Line2_acc_train, SPEC_L2 ,label="Switch 40%",  linewidth = LINEWIDTH)
		if DRAW_Line3:	plt.plot(count_Line3, Line3_acc_train, SPEC_L3 ,label="Switch 60%",  linewidth = LINEWIDTH)
		if DRAW_Line4:	plt.plot(count_Line4, Line4_acc_train, SPEC_L4 ,label="Switch 80%",  linewidth = LINEWIDTH)

		plt.xlabel('# Epochs')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CrossModel_Training_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(5)
		plt.title('Predict Accuracy of Test Set')
		
		if DRAW_Line1: plt.plot(count_Line1, Line1_acc_test, SPEC_L1 ,label="Switch 20%", linewidth = LINEWIDTH)
		if DRAW_Line2: 	plt.plot(count_Line2, Line2_acc_test, SPEC_L2 ,label="Switch 40%",  linewidth = LINEWIDTH)
		if DRAW_Line3: 	plt.plot(count_Line3, Line3_acc_test, SPEC_L3 ,label="Switch 60%", linewidth = LINEWIDTH)
		if DRAW_Line4: 	plt.plot(count_Line4, Line4_acc_test, SPEC_L4 ,label="Switch 80%", linewidth = LINEWIDTH)

		plt.xlabel('# Epochs')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CrossModel_Test_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(6)
		plt.title('Loss of Test Set')
		
		if DRAW_Line1: plt.plot(count_Line1, Line1_loss_test, SPEC_L1 ,label="Switch 20%", linewidth = LINEWIDTH)
		if DRAW_Line2: 	plt.plot(count_Line2, Line2_loss_test, SPEC_L2 ,label="Switch 40%",  linewidth = LINEWIDTH)
		if DRAW_Line3: 	plt.plot(count_Line3, Line3_loss_test, SPEC_L3 ,label="Switch 60%", linewidth = LINEWIDTH)
		if DRAW_Line4: 	plt.plot(count_Line4, Line3_loss_test, SPEC_L4 ,label="Switch 80%", linewidth = LINEWIDTH)

		plt.xlabel('# Epochs')
		plt.ylabel('Loss')
		plt.legend()
		pylab.savefig(PATH_FIGURE+'CrossModel_Test_Set_Loss'+'.png',bbox_inches='tight')
		# plt.show()



		#PLOT Per Second
		matplotlib.rcParams.update({'font.size': 16})
		plt.figure(7)
		plt.title('Loss of Validation Set')
		
		if DRAW_Line1: 	plt.plot(Line1_epoch_times, Line1_loss_val, SPEC_L1 ,label="Switch 20%",  linewidth = LINEWIDTH)
		if DRAW_Line2: 	plt.plot(Line2_epoch_times, Line2_loss_val, SPEC_L2 ,label="Switch 40%",  linewidth = LINEWIDTH)
		if DRAW_Line3:	plt.plot(Line3_epoch_times, Line3_loss_val, SPEC_L3 ,label="Switch 60%",  linewidth = LINEWIDTH)
		if DRAW_Line4:	plt.plot(Line4_epoch_times, Line4_loss_val, SPEC_L4 ,label="Switch 80%",  linewidth = LINEWIDTH)

		plt.xlabel('Seconds')
		plt.ylabel('Loss')
		plt.legend()
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CroszsModel_Validation_Set_Loss'+'.png',bbox_inches='tight')

		plt.figure(8)
		plt.title('Predict Accuracy of Validation Set')
		
		if DRAW_Line1: 	plt.plot(Line1_epoch_times, Line1_acc_val, SPEC_L1 ,label="Switch 20%", linewidth = LINEWIDTH)
		if DRAW_Line2: 	plt.plot(Line2_epoch_times, Line2_acc_val, SPEC_L2 ,label="Switch 40%",  linewidth = LINEWIDTH)
		if DRAW_Line3:	plt.plot(Line3_epoch_times, Line3_acc_val, SPEC_L3 ,label="Switch 60%",  linewidth = LINEWIDTH)
		if DRAW_Line4:	plt.plot(Line4_epoch_times, Line4_acc_val, SPEC_L4 ,label="Switch 80%",  linewidth = LINEWIDTH)

		plt.xlabel('Seconds')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CrossModel_Validation_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(9)
		plt.title('Loss of Training Set')
		# if Y_LIM_FINE_TUNING:	pylab.ylim([-0.01,0.25])
		
		if DRAW_Line1: 	plt.plot(Line1_epoch_times, Line1_loss_train, SPEC_L1 ,label="Switch 20%", linewidth = LINEWIDTH)
		if DRAW_Line2: 	plt.plot(Line2_epoch_times, Line2_loss_train, SPEC_L2 ,label="Switch 40%",  linewidth = LINEWIDTH)
		if DRAW_Line3:	plt.plot(Line3_epoch_times, Line3_loss_train, SPEC_L3 ,label="Switch 60%", linewidth = LINEWIDTH)
		if DRAW_Line4:	plt.plot(Line4_epoch_times, Line4_loss_train, SPEC_L4 ,label="Switch 80%", linewidth = LINEWIDTH)

		plt.xlabel('Seconds')
		plt.ylabel('Loss')
		plt.legend()
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CrossModel_Training_Set_Loss'+'.png',bbox_inches='tight')

		plt.figure(10)
		plt.title('Predict Accuracy of Training Set')
		if Y_LIM_FINE_TUNING:	pylab.ylim([0.93,1.01])
		
		if DRAW_Line1: 	plt.plot(Line1_epoch_times, Line1_acc_train, SPEC_L1 ,label="Switch 20%", linewidth = LINEWIDTH)
		if DRAW_Line2: 	plt.plot(Line2_epoch_times, Line2_acc_train, SPEC_L2 ,label="Switch 40%",  linewidth = LINEWIDTH)
		if DRAW_Line3:	plt.plot(Line3_epoch_times, Line3_acc_train, SPEC_L3 ,label="Switch 60%",  linewidth = LINEWIDTH)
		if DRAW_Line4:	plt.plot(Line4_epoch_times, Line4_acc_train, SPEC_L4 ,label="Switch 80%",  linewidth = LINEWIDTH)

		plt.xlabel('Seconds')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CrossModel_Training_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(11)
		plt.title('Predict Accuracy of Test Set')
		
		if DRAW_Line1: plt.plot(Line1_epoch_times, Line1_acc_test, SPEC_L1 ,label="Switch 20%", linewidth = LINEWIDTH)
		if DRAW_Line2: 	plt.plot(Line2_epoch_times, Line2_acc_test, SPEC_L2 ,label="Switch 40%",  linewidth = LINEWIDTH)
		if DRAW_Line3: 	plt.plot(Line3_epoch_times, Line3_acc_test, SPEC_L3 ,label="Switch 60%", linewidth = LINEWIDTH)
		if DRAW_Line4: 	plt.plot(Line4_epoch_times, Line4_acc_test, SPEC_L4 ,label="Switch 80%", linewidth = LINEWIDTH)

		plt.xlabel('Seconds')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CrossModel_Test_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(12)
		plt.title('Loss of Test Set')
		
		if DRAW_Line1: plt.plot(Line1_epoch_times, Line1_loss_test, SPEC_L1 ,label="Switch 20%", linewidth = LINEWIDTH)
		if DRAW_Line2: 	plt.plot(Line2_epoch_times, Line2_loss_test, SPEC_L2 ,label="Switch 40%",  linewidth = LINEWIDTH)
		if DRAW_Line3: 	plt.plot(Line3_epoch_times, Line3_loss_test, SPEC_L3 ,label="Switch 60%", linewidth = LINEWIDTH)
		if DRAW_Line4: 	plt.plot(Line4_epoch_times, Line4_loss_test, SPEC_L4 ,label="Switch 80%", linewidth = LINEWIDTH)

		plt.xlabel('Seconds')
		plt.ylabel('Loss')
		plt.legend()
		pylab.savefig(PATH_FIGURE+'Time_CrossModel_Test_Set_Loss'+'.png',bbox_inches='tight')
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


