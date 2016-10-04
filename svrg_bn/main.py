import os
import sys

import subprocess
import time

#default value
NUM_EPOCHS = "500"
DEVICE = "GPU1" 
GRADIENT = "svrg"
NUM_HIDDEN_NODES ="500"

def main(num_epochs=NUM_EPOCHS, device = DEVICE, num_hidden_nodes=NUM_HIDDEN_NODES, gradient = GRADIENT):
    device = device.lower()
    str_device = "THEANO_FLAGS=mode=FAST_RUN,device="+device+",floatX=float32 "

    if gradient != "all":
        if (device == "cpu") or ("gpu" in device):
            os.system(str_device + " python classifier_test.py mlpbn "+ gradient + " "+num_epochs+" "+num_hidden_nodes)
    elif gradient == "all":

        methods = ["svrg", "stream", "adagrad"]
        devices = ["GPU0", "GPU1", "GPU2"]
        combos = dict(zip(methods, devices))

        processes = set()
        max_processes = 5
        for method in combos:
            device = combos[method]
	    device = device.lower()
            str_device = "THEANO_FLAGS=mode=FAST_RUN,device="+device+",floatX=float32 "
            command = str_device + " python classifier_test.py mlpbn "+ method + " "+num_epochs+" "+num_hidden_nodes	
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

    os.system("python draw_3_BN.py")


    # else:
    #    print "Unsupported device, please type cpu or gpu"
    # os.system("python draw.py "+num_epochs)

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv) or ('help' in sys.argv):
        print ("Run MLPBN with SVRG or StreamingSVRG or adaGrad:")
        print ("arg:\t[NUM_EPOCHS](500)")
        print ("arg:\t[svrg\stream(OR streaming OR streamingsvrg)\ adagrad\ all(Parallel)](default="+GRADIENT+")")
        print ("arg:\t[cpu\gpu\draw](default="+DEVICE+")")                
        print ("arg:\t[NUM_HIDDEN_NODES](500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = sys.argv[1]
        if len(sys.argv) > 2:
            gradient_name = sys.argv[2]
            if (gradient_name == "streaming" or gradient_name == "streamingsvrg"):
                gradient_name = "stream"
            kwargs['gradient'] = gradient_name            
        if len(sys.argv) > 3:
            kwargs['device'] = sys.argv[3]        
        if len(sys.argv) > 4:
            kwargs['num_hidden_nodes'] = sys.argv[4]
        main(**kwargs)
