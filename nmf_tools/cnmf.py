import scipy.io as sio
import numpy as np
from sklearn.decomposition import NMF

def ComplexDecomposition(X, k=50, shift = 70, paramDict=None):
    '''
    对实部和虚部平移后分别进行分解，后续进行投影得到一个负数的字典矩阵。
    :param X: 待分解的复数矩阵
    :param shift: 在各个维度上进行的平移
    :param paramDict: 一些分解过程中参数的传入。
    示例如下：{
      Winit： 初始的复数权值
      Hinit： 激活系数的初试值
      Wupdate: 迭代过程中是否对W进行更新
    }
    :return:分解后的复数域数据[W, H]
    '''
    m, n = X.shape()
    W = []
    H = []
    Wupdate = True
    if not paramDict or "Winit" not in paramDict.keys():
        W.append(np.random.random([m, k]))
        W.append(np.random.random([m, k]))
    else:
        W = paramDict['Winit']

    if not paramDict or 'Hinit' not in paramDict.keys():
        H.append(np.random.random([k, n]))
        H.append(np.random.random([k, n]))
    else:
        H = paramDict['Hinit']

    if paramDict and 'Wupdate' in paramDict.keys():
        Wupdate = paramDict['Wupdate']

    X_complex = [np.real(X), np.imag(X)]
    for x in X_complex:
