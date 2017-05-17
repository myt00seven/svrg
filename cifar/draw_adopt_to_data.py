# Draw for NI dataset

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
FIND_BEST = True 
# output the best predict when loss on validation is lowest
DRAW_Line1 = True
LOAD_SVRG = True

DRAW_Line1 = True
DRAW_Line2 = True
DRAW_Line3 = True
DRAW_Line4 = True

Y_LIM_FINE_TUNING = True

MODE = "all"

PRINT = [
0, #fill the index 0
1, #'NI_CM_Validation_Loss'                     1
1, #'NI_CM_Validation_Set_Accuracy'                     2
0, #'NI_CM_Training_Set_Loss'                     3
0, #'NI_CM_Training_Set_Accuracy'                     4
0, #'NI_CM_Test_Set_Accuracy'                     5
0, #'NI_CM_Test_Loss'                     6
0, #'Timediff_Validation_Loss'                     7
0, #'TimeDiff_Validation_Set_Accuracy'                     8
0, #'TimeDiff_Training_Set_Loss'                     9
0, #'TimeDiff_Training_Set_Accuracy'                     10
0, #'TimeDiff_Test_Set_Accuracy'                     11
0, #'TimeDiff_Test_Loss'                     12
1, #'NI_CM_Validation_Set_Error'                     13
]


styles_11_colors = [
('r', '-'), # 1
('b', '-'), # 0.75
('g', '-'), # 0.5
('y', '-'), # 0.25
('c', '--'),    # 0.1
('m', '--'),    # 0.01
('b', '--'),    # 0.001
('b', ':'), # 1/m
('c', ':'), # 1/m^2
('k', '-')  # 0
]

styles_5_colors = [
('r', '-'), # 1
('g', '-'), # 0.5
('m', '--'),    # 0.01
('b', ':'), # 1/m
('c', ':'), # 1/m^2
]

# Number of Moving Average

# SPEC_L1 = 'bo-'
# SPEC_L1 = 'g^--'
# SPEC_L2 = 'cs:'
# SPEC_L4 = 'r*-.'


# SPEC = ['b-', 'c:', 'r-.', 'g--', 'kv', 'c-', 'r:', 'g-.', 'k-','r-', 'g:','rv','gv']



def my_min_index(sequence):
    """return the index of the minimum element of sequence"""
    low = sequence[0] # need to start with some value
    low_index = 0
    idx = 0
    for i in sequence:
        if i < low:
            low = i
            low_index = idx
        idx = idx + 1
    return low_index

NUM_EPOCHS = 10

def main(num_epochs=NUM_EPOCHS, mode = MODE):

    if mode == 'all':
        styles = styles_11_colors
    if mode == 'select' or mode == "single":
        styles = styles_5_colors

    tag_first_line=True
    models= []
    models_name= []

    if mode == "all":
        models_file_name = 'models.txt'
    elif mode == "select":
        models_file_name = "select_models.txt"
    elif mode == "single":
        models_file_name = "single_models.txt"
    else:
        colors=('m','c','b','g','r')
        linestyles=('-','--','-.',':')
        styles=[(color,linestyle) for linestyle in linestyles for color in colors]
        random.shuffle(styles)
        # print styles

    with open(models_file_name) as f:
        for line in f:
            if tag_first_line:
                models_count = int(line)
                tag_first_line=False
            else:
            	data = line.split(",")
                models.append(str(data[0]))
                models_name.append(str(data[1]))


    line_acc_test = []
    line_acc_test=[]
    line_acc_train=[]
    line_acc_val=[]
    line_err_val=[]
    line_loss_test=[]
    line_loss_train=[]
    line_loss_val=[]
    line_epoch_times=[]
    count_line=[]

    for model in models:
        # print str(model)
        # print model
        index = models.index(model)
        # line_acc_test.append(np.loadtxt(PATH_DATA+str(model)+"_acc_test.txt"))
        # line_acc_train.append(np.loadtxt(PATH_DATA+str(model)+"_acc_train.txt"))
        line_acc_val.append(np.loadtxt(PATH_DATA+str(model)+"_acc_val.txt"))
        line_err_val.append(np.loadtxt(PATH_DATA+str(model)+"_err_val.txt"))
        # line_loss_test.append(np.loadtxt(PATH_DATA+str(model)+"_loss_test.txt"))
        # line_loss_train.append(np.loadtxt(PATH_DATA+str(model)+"_loss_train.txt"))
        line_loss_val.append(np.loadtxt(PATH_DATA+str(model)+"_loss_val.txt"))
        # line_epoch_times.append(np.loadtxt(PATH_DATA+str(model)+"_epoch_times.txt"))
        count_line.append(np.arange(line_acc_val[index].shape[0])+1)
        # print count_line
        
    str_epochs = str(num_epochs)
    MAXLENGTH = num_epochs
    

    # if (MAXLENGTH>0 or STARTPOINT>0):        # Need add for epoch_times
    #     for model in models:
    #         index = models.index(model)
    #         line_acc_test[index]    =     line_acc_test[STARTPOINT:MAXLENGTH+1]
    #         line_acc_train[index]    =     line_acc_train[STARTPOINT:MAXLENGTH+1]
    #         line_acc_val[index]        =     line_acc_val[STARTPOINT:MAXLENGTH+1]
    #         line_loss_test[index]    =     line_loss_test[STARTPOINT:MAXLENGTH+1]
    #         line_loss_train[index]    =     line_loss_train[STARTPOINT:MAXLENGTH+1]
    #         line_loss_val[index]    =     line_loss_val[STARTPOINT:MAXLENGTH+1]
    #         line_epoch_times[index]    =     line_epoch_times[STARTPOINT:MAXLENGTH+1]
    #         count_line[index]        =     count_line[STARTPOINT:MAXLENGTH+1]



    if DRAW_COMPARE:

        #PLFOT 
        matplotlib.rcParams.update({'font.size': 16})

#!!!
        if PRINT[1] == 1:
            plt.figure(1)
            plt.title('Loss on Validation')

            
            for model in models:
                index = models.index(model)
                # print line_loss_val[index]
                plt.plot(count_line[index], line_loss_val[index], color=styles[index][0],ls=styles[index][1] ,label=models_name[index],  linewidth = LINEWIDTH)

            plt.xlabel('# Epochs')
            plt.ylabel('Loss')
            # plt.legend(bbox_to_anchor=(1,0.65)) # 10 lines
            # plt.legend(bbox_to_anchor=(1,1)) # 5 lines
            plt.legend(loc='best', fontsize = 12)
            axes = plt.gca()
            if mode == 'select':
                axes.set_ylim([0.0,1]) # 5 lines

            # plt.show()
            pylab.savefig(PATH_FIGURE+'NI_CM_Validation_Loss'+'.png',bbox_inches='tight')

#!!1
        if PRINT[2] == 1:
            plt.figure(2)
            plt.title('Accuracy on Validation')
            
            if mode == 'select':
                axes.set_ylim([0.915,0.93]) # 5 line

            for model in models:
                index = models.index(model)                
                plt.plot(count_line[index], line_acc_val[index], color=styles[index][0],ls=styles[index][1] ,label=models_name[index], linewidth = LINEWIDTH)
            

            plt.xlabel('# Epochs')
            plt.ylabel('Predict Accuracy')
            plt.legend(loc='best', fontsize = 12)
            
            # plt.show()
            pylab.savefig(PATH_FIGURE+'NI_CM_Validation_Accuracy'+'.png',bbox_inches='tight')

        if PRINT[3] == 1:
            plt.figure(3)
            plt.title('Loss on Training')
            # if Y_LIM_FINE_TUNING:    pylab.ylim([-0.01,0.25])    
            for model in models:
                index = models.index(model)
                plt.plot(count_line[index], line_loss_train[index], color=styles[index][0],ls=styles[index][1] ,label=models_name[index], linewidth = LINEWIDTH)
            

            plt.xlabel('# Epochs')
            plt.ylabel('Loss')
            plt.legend()
            # plt.show()
            pylab.savefig(PATH_FIGURE+'NI_CM_Training_Set_Loss'+'.png',bbox_inches='tight')

        if PRINT[4] == 1:
            plt.figure(4)
            plt.title('Accuracy on Training')
            if Y_LIM_FINE_TUNING:    pylab.ylim([0.93,1.01])    
            for model in models:
                index = models.index(model)
                plt.plot(count_line[index], line_acc_train[index], color=styles[index][0],ls=styles[index][1] ,label=models_name[index], linewidth = LINEWIDTH)
            

            plt.xlabel('# Epochs')
            plt.ylabel('Predict Accuracy')
            plt.legend(bbox_to_anchor=(1,0.4))
            # plt.show()
            pylab.savefig(PATH_FIGURE+'NI_CM_Training_Set_Accuracy'+'.png',bbox_inches='tight')

        if PRINT[5] == 1:
            plt.figure(5)
            plt.title('Accuracy on Test')

            axes = plt.gca()
            # axes.set_xlim([xmin,xmax])
            if mode == 'select':
                axes.set_ylim([0.915,0.93]) # 5 line
            
            for model in models:
                index = models.index(model)
                plt.plot(count_line[index], line_acc_test[index], color=styles[index][0],ls=styles[index][1] ,label=models_name[index], linewidth = LINEWIDTH)

            plt.xlabel('# Epochs')
            plt.ylabel('Predict Accuracy')
            # plt.legend(bbox_to_anchor=(1,0.8)) # all 10 lines
            # plt.legend(bbox_to_anchor=(1,0.4)) # 5 line
            plt.legend(loc='best', fontsize = 12)

            # plt.show()
            pylab.savefig(PATH_FIGURE+'NI_CM_Test_Set_Accuracy'+'.png',bbox_inches='tight')

#!!!
        if PRINT[13] == 1:
            plt.figure(13)
            plt.title('Error on Validation')
            
            if mode == 'select':
                axes.set_ylim([0.,0.3]) # 5 line

            for model in models:
                index = models.index(model)                
                plt.plot(count_line[index], line_err_val[index], color=styles[index][0],ls=styles[index][1] ,label=models_name[index], linewidth = LINEWIDTH)

            plt.xlabel('# Epochs')
            plt.ylabel('Test Error')
            plt.legend(loc='best', fontsize = 12)
            
            # plt.show()
            pylab.savefig(PATH_FIGURE+'NI_CM_Validation_Error'+'.png',bbox_inches='tight') 

    if FIND_BEST == True:

        o_file = open(PATH_FIGURE+'best_record.txt', 'w+')

        o_file.write("Min_Epoch \t Loss_Val \t Accuracy_Test \t Model\n")

        for model in models:

            index = models.index(model)
            index_min = my_min_index(line_loss_val[index])
            o_file.write("%d\t\t" %index_min)
            o_file.write("%.4f\t\t" %line_loss_val[index][index_min])
            o_file.write("%.4f\t\t" %line_acc_val[index][index_min])
            o_file.write(models_name[index]+ "\n")

            


    
    print ("Finish drawing cross model plots.")

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv) or ('help' in sys.argv):
        print("arg: NUM_EPOCHS")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            if sys.argv[1] == "select":
                kwargs['mode'] = "select"
            elif sys.argv[1] == "single":
                kwargs['mode'] = "single"
            else:
                kwargs['mode'] = "all"
                kwargs['num_epochs'] = int(sys.argv[1])
        main(**kwargs)


