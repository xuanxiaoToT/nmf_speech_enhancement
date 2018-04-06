import numpy as np
from numpy import diag
from numpy import sqrt

"""
这个文件定义了与MATLAB相同的函数，这些函数可以在转换m文件时，减少工作量。`
"""

def iscell(arr):
    return isinstance(arr, list)

def isfield(dic, key):
    return key in dic.keys()

def isempty(dic, key):
    return not isfield(dic, key) or not dic[key]

def cell(s, l):
    return [[None for _ in range(s)] for _ in range(l)]

def rand(m, n, l=None):
    if l:
        return [np.random.random([m, n]) for _ in range(l)]
    else:
        return np.random.random([m, n])
def max(a, ax = None):
    return np.max(a, axis=ax)

def length(param):
    return len(param)

def num2str(i):
    return str(i)

def error(lis):
    raise Exception("".join(lis))

eps = 0.00001

false = False
true = True