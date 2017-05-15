# This is used to draw three comparisons for SGD+BN, SVRG+BN and Streaming SVRG +BN

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg') 
import pylab
import numpy as np
import random
		
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

PATH_DATA   = "data_large/"

PATH_FIGURE = "figure_arbitrary/"

STARTPOINT = 0
LINEWIDTH = 2

DRAW_COMPARE = True
DRAW_Line1 = True
LOAD_SVRG = True

DRAW_Line1 = True
DRAW_Line2 = True
DRAW_Line3 = True
DRAW_Line4 = True

Y_LIM_FINE_TUNING = True

# Number of Moving Average

# SPEC_L1 = 'bo-'
# SPEC_L1 = 'g^--'
# SPEC_L2 = 'cs:'
# SPEC_L4 = 'r*-.'


# SPEC = ['b-', 'c:', 'r-.', 'g--', 'kv', 'c-', 'r:', 'g-.', 'k-','r-', 'g:','rv','gv']

colors=('m','c','b','g','r')
#,'#CE0058' Rubine
# k black
# y yellow
linestyles=('-','--','-.',':')
styles=[(color,linestyle) for linestyle in linestyles for color in colors]

# print styles
random.shuffle(styles)
# print styles


NUM_EPOCHS = 10

def main(num_epochs=NUM_EPOCHS):

	tag_first_line=True
	models= []
	with open(PATH_DATA+'models.txt') as f:
		for line in f:
			data = line.split()
			if tag_first_line:
				models_count = int(line)
				tag_first_line=False
			else:
				models.append(str(data[0]))


	line_acc_test = []
	line_acc_test=[]
	line_acc_train=[]
	line_acc_val=[]
	line_loss_test=[]
	line_loss_train=[]
	line_loss_val=[]
	line_epoch_times=[]
	count_line=[]

	for model in models:
		# print str(model)
		# print model
		index = models.index(model)
		line_acc_test.append(np.loadtxt(PATH_DATA+str(model)+"_acc_test.txt"))
		line_acc_train.append(np.loadtxt(PATH_DATA+str(model)+"_acc_train.txt"))
		line_acc_val.append(np.loadtxt(PATH_DATA+str(model)+"_acc_val.txt"))
		line_loss_test.append(np.loadtxt(PATH_DATA+str(model)+"_loss_test.txt"))
		line_loss_train.append(np.loadtxt(PATH_DATA+str(model)+"_loss_train.txt"))
		line_loss_val.append(np.loadtxt(PATH_DATA+str(model)+"_loss_val.txt"))
		line_epoch_times.append(np.loadtxt(PATH_DATA+str(model)+"_epoch_times.txt"))
		count_line.append(np.arange(line_acc_val[index].shape[0])+1)
		# print count_line
		
	str_epochs = str(num_epochs)
	MAXLENGTH = num_epochs
	

	# if (MAXLENGTH>0 or STARTPOINT>0):		# Need add for epoch_times
	# 	for model in models:
	# 		index = models.index(model)
	# 		line_acc_test[index]	= 	line_acc_test[STARTPOINT:MAXLENGTH+1]
	# 		line_acc_train[index]	= 	line_acc_train[STARTPOINT:MAXLENGTH+1]
	# 		line_acc_val[index]		= 	line_acc_val[STARTPOINT:MAXLENGTH+1]
	# 		line_loss_test[index]	= 	line_loss_test[STARTPOINT:MAXLENGTH+1]
	# 		line_loss_train[index]	= 	line_loss_train[STARTPOINT:MAXLENGTH+1]
	# 		line_loss_val[index]	= 	line_loss_val[STARTPOINT:MAXLENGTH+1]
	# 		line_epoch_times[index]	= 	line_epoch_times[STARTPOINT:MAXLENGTH+1]
	# 		count_line[index]		= 	count_line[STARTPOINT:MAXLENGTH+1]



	if DRAW_COMPARE:

		#PLFOT 
		matplotlib.rcParams.update({'font.size': 15})
		plt.figure(1)
		plt.title('Loss of Validation Set')
		
		for model in models:
			index = models.index(model)
			plt.plot(count_line[index], line_loss_val[index], color=styles[index][0],ls=styles[index][1] ,label=model,  linewidth = LINEWIDTH)

		plt.xlabel('# Epochs')
		plt.ylabel('Loss')
		plt.legend()
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CroszsModel_Validation_Set_Loss'+'.png',bbox_inches='tight')

		plt.figure(2)
		plt.title('Predict Accuracy of Validation Set')
		
		for model in models:
			index = models.index(model)
			plt.plot(count_line[index], line_acc_val[index], color=styles[index][0],ls=styles[index][1] ,label=model, linewidth = LINEWIDTH)
		

		plt.xlabel('# Epochs')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CrossModel_Validation_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(3)
		plt.title('Loss of Training Set')
		# if Y_LIM_FINE_TUNING:	pylab.ylim([-0.01,0.25])	
		for model in models:
			index = models.index(model)
			plt.plot(count_line[index], line_loss_train[index], color=styles[index][0],ls=styles[index][1] ,label=model, linewidth = LINEWIDTH)
		

		plt.xlabel('# Epochs')
		plt.ylabel('Loss')
		plt.legend()
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CrossModel_Training_Set_Loss'+'.png',bbox_inches='tight')

		plt.figure(4)
		plt.title('Predict Accuracy of Training Set')
		if Y_LIM_FINE_TUNING:	pylab.ylim([0.93,1.01])	
		for model in models:
			index = models.index(model)
			plt.plot(count_line[index], line_acc_train[index], color=styles[index][0],ls=styles[index][1] ,label=model, linewidth = LINEWIDTH)
		

		plt.xlabel('# Epochs')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CrossModel_Training_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(5)
		plt.title('Predict Accuracy of Test Set')
		
		for model in models:
			index = models.index(model)
			plt.plot(count_line[index], line_acc_test[index], color=styles[index][0],ls=styles[index][1] ,label=model, linewidth = LINEWIDTH)
		

		plt.xlabel('# Epochs')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'CrossModel_Test_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(6)
		plt.title('Loss of Test Set')
		
		for model in models:
			index = models.index(model)
			plt.plot(count_line[index], line_loss_test[index], color=styles[index][0],ls=styles[index][1] ,label=model, linewidth = LINEWIDTH)
		

		plt.xlabel('# Epochs')
		plt.ylabel('Loss')
		plt.legend()
		pylab.savefig(PATH_FIGURE+'CrossModel_Test_Set_Loss'+'.png',bbox_inches='tight')
		# plt.show()

		#PLOT Per Second
		matplotlib.rcParams.update({'font.size': 16})
		plt.figure(7)
		plt.title('Loss of Validation Set')
		
		for model in models:
			index = models.index(model)
			plt.plot(line_epoch_times[index], line_loss_val[index], color=styles[index][0],ls=styles[index][1] ,label=model,  linewidth = LINEWIDTH)
		

		plt.xlabel('Seconds')
		plt.ylabel('Loss')
		plt.legend()
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CroszsModel_Validation_Set_Loss'+'.png',bbox_inches='tight')

		plt.figure(8)
		plt.title('Predict Accuracy of Validation Set')
		
		for model in models:
			index = models.index(model)
			plt.plot(line_epoch_times[index], line_acc_val[index], color=styles[index][0],ls=styles[index][1] ,label=model, linewidth = LINEWIDTH)
		

		plt.xlabel('Seconds')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CrossModel_Validation_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(9)
		plt.title('Loss of Training Set')
		# if Y_LIM_FINE_TUNING:	pylab.ylim([-0.01,0.25])
		
		for model in models:
			index = models.index(model)
			plt.plot(line_epoch_times[index], line_loss_train[index], color=styles[index][0],ls=styles[index][1] ,label=model, linewidth = LINEWIDTH)
		

		plt.xlabel('Seconds')
		plt.ylabel('Loss')
		plt.legend()
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CrossModel_Training_Set_Loss'+'.png',bbox_inches='tight')

		plt.figure(10)
		plt.title('Predict Accuracy of Training Set')
		if Y_LIM_FINE_TUNING:	pylab.ylim([0.93,1.01])
		
		for model in models:
			index = models.index(model)
			plt.plot(line_epoch_times[index], line_acc_train[index], color=styles[index][0],ls=styles[index][1] ,label=model, linewidth = LINEWIDTH)
		

		plt.xlabel('Seconds')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CrossModel_Training_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(11)
		plt.title('Predict Accuracy of Test Set')
		
		for model in models:
			index = models.index(model)
			plt.plot(line_epoch_times[index], line_acc_test[index], color=styles[index][0],ls=styles[index][1] ,label=model, linewidth = LINEWIDTH)
		

		plt.xlabel('Seconds')
		plt.ylabel('Predict Accuracy')
		plt.legend(bbox_to_anchor=(1,0.4))
		# plt.show()
		pylab.savefig(PATH_FIGURE+'Time_CrossModel_Test_Set_Predict_Accuracy'+'.png',bbox_inches='tight')

		plt.figure(12)
		plt.title('Loss of Test Set')
		
		for model in models:
			index = models.index(model)
			plt.plot(line_epoch_times[index], line_loss_test[index], color=styles[index][0],ls=styles[index][1] ,label=model, linewidth = LINEWIDTH)
		

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


