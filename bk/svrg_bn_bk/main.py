import os
import sys

#default value
NUM_EPOCHS = "500"
DEVICE = "GPU1" 
GRADIENT = "svrg"
NUM_HIDDEN_NODES ="500"

def main(num_epochs=NUM_EPOCHS, device = DEVICE, num_hidden_nodes=NUM_HIDDEN_NODES, gradient = GRADIENT):
    device = device.lower()
    str_device = "THEANO_FLAGS=mode=FAST_RUN,device="+device+",floatX=float32 "
    if (device == "cpu") or ("gpu" in device):
        os.system(str_device + " python classifier_test.py mlpbn "+ gradient + " "+num_epochs+" "+num_hidden_nodes)
    # else:
    #    print "Unsupported device, please type cpu or gpu"
    # os.system("python draw.py "+num_epochs)

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv) or ('help' in sys.argv):
        print ("Run MLPBN with SVRG or StreamingSVRG or adaGrad:")
        print ("arg:\t[cpu\gpu\draw](default="+DEVICE+")")
        print ("arg:\t[svrg\stream(OR streaming OR streamingsvrg)\ adagrad](default="+GRADIENT+")")
        print ("arg:\t[NUM_EPOCHS](500)")
        print ("arg:\t[NUM_HIDDEN_NODES](500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['device'] = sys.argv[1]
        if len(sys.argv) > 2:
            gradient_name = sys.argv[2]
            if (gradient_name == "streaming" or gradient_name == "streamingsvrg"):
                gradient_name = "stream"
            kwargs['gradient'] = gradient_name
        if len(sys.argv) > 3:
            kwargs['num_epochs'] = sys.argv[3]
        if len(sys.argv) > 4:
            kwargs['num_hidden_nodes'] = sys.argv[4]
        main(**kwargs)
