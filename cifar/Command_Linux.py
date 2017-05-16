Hints:
 python main.py --help
Run MLP and MLPBN with given parameters:
arg:	[NUM_EPOCHS](500)
	[NUM_HIDDEN_NODES](500)
	[cpu\gpu\draw](default=GPU1)
	[original\dbn\dbn2\const?](What BN lagorithm to use, default=original)
		(original is the original BN algorithm. dbn is 1/m MA, dbn2 is 1/m^2 MA.)
		(const_? is constant alpha=?,?=1, 075, 05, 025, 01, 001, 0001, 0

================================================================================================================

# commands for CIFAR dataset
# the num_hidden_nodes is invalidate here

python main.py 50 50 gpu0 const1
python main.py 50 50 gpu1 const075
python main.py 50 50 gpu2 const05
python main.py 50 50 gpu1 const025
python main.py 50 50 gpu2 const01
python main.py 50 50 gpu1 const001
python main.py 50 50 gpu2 const0001
python main.py 50 50 gpu1 dbn
python main.py 50 50 gpu2 dbn2
python main.py 50 50 gpu1 const0
python main.py 50 300 gpu2 const01

# this is for MNIST
# I miss this run


THEANO_FLAGS=mode=FAST_RUN,device=gpu2 python cifar_cnn.py