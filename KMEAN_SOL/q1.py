import numpy as np
from scipy import linalg

M = np.matrix([[1, 2], [2, 1], [3, 4], [4, 3]])
U, s, Vh = linalg.svd(M, full_matrices=False)
Evals, Evecs = linalg.eigh(M.transpose()*M)
sortIndex = Evals.argsort()[::-1]
Evals = Evals[sortIndex]
Evecs = Evecs[:,sortIndex]