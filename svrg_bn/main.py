import os
import sys

import subprocess
import time

#default value
NUM_EPOCHS = "5"
DEVICE = "cpu"
GRADIENT = "adagrad"
NUM_HIDDEN_NODES ="500"
NUM_REP = "1"
IF_SWITCH = 0

methods = ["svrg", "streaming", "adagrad"]

def main(num_epochs=NUM_EPOCHS, device = DEVICE, num_hidden_nodes=NUM_HIDDEN_NODES, gradient = GRADIENT, num_rep = NUM_REP, if_switch = IF_SWITCH):
    device = device.lower()
    str_device = "THEANO_FLAGS=mode=FAST_RUN,device="+device+",floatX=float32 "
    
    for each in methods:
        file_clean = open("data/best_result_"+each+".txt",'w')
        file_clean.close()

    for loop_idx in range(0,int(num_rep)):
        if gradient != "all":
            if (device == "cpu") or ("gpu" in device):
                os.system(str_device + " python classifier_test.py mlpbn "+ gradient + " "+num_epochs+" "+if_switch+" "+num_hidden_nodes)
        elif gradient == "all":
            devices = ["GPU0", "GPU1", "GPU2"]
            combos = dict(zip(methods, devices))

            processes = set()
            max_processes = 5
            for method in combos:
                device = combos[method]
                device = device.lower()
                str_device = "THEANO_FLAGS=mode=FAST_RUN,device="+device+",floatX=float32 "
                command = str_device + " python classifier_test.py mlpbn "+ method + " "+num_epochs+" "+if_switch+" "+num_hidden_nodes
                #command = 'help'
                print(command)

                processes.add(subprocess.Popen(command , shell = True))
                if len(processes) >= max_processes:
                    os.wait()
                    processes.difference_update(
                        [p for p in processes if p.poll() is not None])
            #Check if all the child processes were closed
            for p in processes:
                if p.poll() is None:
                    p.wait()

        os.system("python draw_3_BN.py " + num_epochs)


    # else:
    #    print "Unsupported device, please type cpu or gpu"
    # os.system("python draw.py "+num_epochs)

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv) or ('help' in sys.argv):
        print ("Run MLPBN with SVRG or StreamingSVRG or adaGrad:")
        print ("arg:\t[NUM_Rep] (1) Repeat entire program for how many times")
        print ("arg:\t[NUM_EPOCHS](500)")
        print ("arg:\t[svrg\stream(OR streaming OR streamingsvrg OR ssvrg)\ adagrad\ all(Parallel)](default="+GRADIENT+")")
        print ("arg:\t[cpu\gpu\draw](default="+DEVICE+")")                
        print ("arg:\t[y \ n\ 1\ 0](If switch method, default is adagrad to ssvrg")                
        print ("arg:\t[NUM_HIDDEN_NODES](500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_rep'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = sys.argv[2]
        if len(sys.argv) > 3:
            gradient_name = sys.argv[3]
            if (gradient_name == "stream" or gradient_name == "streamingsvrg" or gradient_name == "ssvrg"):
                gradient_name = "streaming"
            kwargs['gradient'] = gradient_name            
        if len(sys.argv) > 4:
            kwargs['device'] = sys.argv[4]
        if len(sys.argv) > 5:
            tag_switch = sys.argv[5]
            if (tag_switch == "y" or tag_switch == "yes" or tag_switch == "1"):
                tag_switch = 1
            else:
                tag_switch = 0
            kwargs['if_switch'] = sys.argv[5]
        if len(sys.argv) > 6:
            kwargs['num_hidden_nodes'] = sys.argv[6]
        main(**kwargs)
