# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: decompose.py
# @Blog    ：http://meepoljd.com

import numpy as np
from sklearn.decomposition import NMF


def decompose(spec, k, max_iter=200000, alpha=0.8, l1_rate=1):
    """
    basic NMF tool, use it to get W and H

    Example:
    >>> V = 10 * np.random.rand(100, 3000)
    >>> W, H = decompose(V, 50)

    :param spec:
    :param k:
    :param max_iter:
    :param alpha:
    :param l1_rate:
    """
    model = NMF(n_components=k, solver='mu', max_iter=max_iter,
                beta_loss='kullback-leibler', alpha=alpha, l1_ratio=l1_rate)
    _dic = model.fit_transform(spec)
    _act = model.components_

    return _dic, _act


def decompose_with_dict(spec, dic, max_iter=200000, alpha=0.8, l1_rate=1):
    """
    get H with V and W

    Example:
    >>> V = 10*np.random.rand(100, 200)
    >>> W, H = decompose(V, k=50)
    >>> H2 = decompose_with_dict(V, W)

    :param spec:
    :param dic:
    :param max_iter
    :param alpha
    :param l1_rate
    :return:
    """
    k = dic.shape[1]
    model = NMF(n_components=k, solver='mu', max_iter=max_iter,
                beta_loss='kullback-leibler', alpha=alpha, l1_ratio=l1_rate)
    model.fit(spec.T)
    model.components_ = dic.T
    _act = model.fit_transform(spec.T)

    return _act.T


if __name__ == '__main__':
    import doctest
    doctest.testmod()
