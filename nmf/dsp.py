# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: dsp.py
# @Blog    ：http://meepoljd.com

import numpy as np
from config import *


# 预加重操作，这里不好用  PESQ下降0.1
def pre_emphasise(signal, rate=emphasis_rate):
    """
    进行信号预加重，增强辅音（高频）部分
    Example：

    :param signal:
    :return:
    """
    return np.array(np.append(signal[0], signal[1:] - rate * signal[:-1]))


def de_emphasise(signal, rate=emphasis_rate):
    """
    去加重操作，还原预加重信号

    Example:


    :param signal:
    :param rate:
    :return:
    """
    res = np.array(signal)
    for i in range(1, len(signal)):
        res[i] += rate * res[i-1]
    return res


# mfcc特征
def mfcc():
    pass


# 对数谱



# 能量谱



if __name__ == '__main__':
    import doctest
    doctest.testmod()