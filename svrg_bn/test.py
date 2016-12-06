import numpy as np

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def deminish(a) :
    end_factor=0.5
    length = len(a)
    for i in range(length):
        a[i] = a[i] * (1-(1-end_factor)*(i/float(length)))
    return a


# a = np.arange(20)
a=[1,1,1,1,1,1,1,1,1]
# a=[1,-1,1,-1,1]
print a
# print moving_average(a, n=3)
# print np.abs(a)
print deminish(a)

