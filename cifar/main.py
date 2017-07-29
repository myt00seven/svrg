import os
import sys
"""
A script the run the program automatically and switch the parameters
run the python script of DL
"""


#default value
NUM_EPOCHS = "500"
DEVICE = "cuda1" 
NUM_HIDDEN_NODES ="500"
BNALG = "original"
MODE = "manual"
devices = ["cuda0", "cuda1", "cuda2"]
 

def main(num_epochs=NUM_EPOCHS, device = DEVICE, num_hidden_nodes=NUM_HIDDEN_NODES, bnalg = BNALG, mode = MODE):
	device = device.lower()
	bnalg = bnalg.lower()
	str_COM1 = "THEANO_FLAGS=mode=FAST_RUN,device="
	str_COM2 = ",floatX=float32 "
	str_device = str_COM1+device+str_COM2

	# if mode == "manual":
	if (device == "cpu") or ("gpu" in device):
		# os.system(str_device + " python large_gpu_mnist.py mlp sgd "+num_epochs+" "+num_hidden_nodes)
		os.system(str_device + " python cifar_cnn.py mlpbn sgd_adagrad "+num_epochs+" "+num_hidden_nodes + " "+ bnalg)
	# elif mode == "all":
		


if __name__ == '__main__':
	if ('--help' in sys.argv) or ('-h' in sys.argv) or ('help' in sys.argv):
		print ("Run MLP and MLPBN with given parameters:")
		print ("arg:\t[NUM_EPOCHS](500)")
		print ("\t[NUM_HIDDEN_NODES](500)")
		print ("\t[cpu\gpu\draw](default="+DEVICE+")")
		print ("\t[original\dbn\dbn2\const?](What BN lagorithm to use, default="+BNALG+")")
		print ("\t\t(original is the original BN algorithm. dbn is 1/m MA, dbn2 is 1/m^2 MA.)")
		print ("\t\t(const_? is constant alpha=?,?=1, 075, 05, 025, 01, 001, 0001, 0")
	else:
		kwargs = {}
		if len(sys.argv) > 1:
			kwargs['num_epochs'] = sys.argv[1]
		if len(sys.argv) > 2:
			kwargs['num_hidden_nodes'] = sys.argv[2]
		if len(sys.argv) > 3:
			kwargs['device'] = sys.argv[3]
		if len(sys.argv) > 4:
			kwargs['bnalg'] = sys.argv[4]
			if kwargs['bnalg'] == "ori":
				kwargs['bnalg'] = "original"
		main(**kwargs)
