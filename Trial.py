import numpy as np

A=np.array([1,3,5])
print(np.tensordot(A, A, axes=0))