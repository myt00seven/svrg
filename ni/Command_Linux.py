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

commands

python main.py 100 50 gpu0 const1
python main.py 100 50 gpu1 const075
python main.py 100 50 gpu2 const05

python main.py 100 50 gpu0 const025
python main.py 100 50 gpu1 const01
python main.py 100 50 gpu2 const001

python main.py 100 50 gpu0 const0001
python main.py 100 50 gpu1 dbn
python main.py 100 50 gpu2 dbn2

python main.py 100 50 gpu2 const0