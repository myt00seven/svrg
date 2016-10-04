import os
import sys
"""
A script the run the program automatically and switch the parameters
run the python script of DL
"""


#default value
NUM_EPOCHS = "500"
DEVICE = "GPU1" 
NUM_HIDDEN_NODES ="500"

def main(num_epochs=NUM_EPOCHS, device = DEVICE, num_hidden_nodes=NUM_HIDDEN_NODES):
	device = device.lower()
	str_device = "THEANO_FLAGS=mode=FAST_RUN,device="+device+",floatX=float32 "
	if (device == "cpu") or ("gpu" in device):
		os.system(str_device + " python large_gpu_mnist.py mlp sgd "+num_epochs+" "+num_hidden_nodes)
		os.system(str_device + " python large_gpu_mnist.py mlpbn sgd "+num_epochs+" "+num_hidden_nodes)
	# else:
	#	 print "Unsupported device, please type cpu or gpu"
	os.system("python draw.py "+num_epochs)

if __name__ == '__main__':
	if ('--help' in sys.argv) or ('-h' in sys.argv) or ('help' in sys.argv):
		print ("Run MLP and MLPBN with given parameters:")
		print ("arg:\t[NUM_EPOCHS](500)")
		print ("\t[NUM_HIDDEN_NODES](500)")
		print ("\t[cpu\gpu\draw](default="+DEVICE+")")
	else:
		kwargs = {}
		if len(sys.argv) > 1:
			kwargs['num_epochs'] = sys.argv[1]
		if len(sys.argv) > 2:
			kwargs['num_hidden_nodes'] = sys.argv[2]
		if len(sys.argv) > 3:
			kwargs['device'] = sys.argv[3]
		main(**kwargs)
