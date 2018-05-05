from nmf_tools import cnmf_euc
from config import Rank, Shift
from sklearn.decomposition import NMF
from DSP import *
from fileManager import find_files

def move2positive(mat, step=Shift):
    tmp = mat + step * (1 + 1j)
    return tmp

def move_back(mat, step=Shift):
    return mat - step * (1 + 1j)

V1 = getSpec('s1/drums.wav')
Vp1 = move2positive(V1)
W1, H1, Vrec1 = cnmf_euc(Vp1, Rank, n_iter=10000)

print()
