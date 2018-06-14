# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: nmf.py
# @Blog    ：http://meepoljd.com

from . import *


def nmf(V, k=rank, maxiter=max_iter, alpha=0, l1_rate=0):
    """
    basic NMF tool, use it to get W and H

    Example:
    >>> V = 10 * np.random.rand(100, 3000)
    >>> W, H, _ = nmf(V, rank)

    :param V:
    :param k:
    :param maxiter:
    """
    model = NMF(n_components=k, solver='mu', max_iter=maxiter,
                beta_loss='kullback-leibler', alpha=alpha, l1_ratio=l1_rate)
    W = model.fit_transform(V)
    H = model.components_
    F = model.reconstruction_err_

    return W, H, F


def nmf_with_W(V, W):
    """
    get H with V and W

    Example:
    >>> V = 10*np.random.rand(100, 200)
    >>> W, H, _ = nmf(V, k=rank)
    >>> H2, err = nmf_with_W(V, W)

    :param V:
    :param W:
    :param maxiter:
    :return:
    """
    H, err, _, _ = np.linalg.lstsq(W, V)
    err = np.mean(err)
    return H, err
