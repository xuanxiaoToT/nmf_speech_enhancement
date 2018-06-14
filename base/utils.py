# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: utils.py
# @Blog    ：http://meepoljd.com


from . import *

def get_spec(signal, sr=sample_rate):
    """
    wrapped the stft in librosa to use the same param
    fs=sample_rate, win_len=32ms, hop=win_len/2

    Example:

    >>> sig, _ = librosa.load('speech/test/fjcs0_sx409.wav', sr=sample_rate)
    >>> spec = get_spec(sig)
    >>> spec.shape[0]==(sample_rate//1000*16)+1
    True
    >>> spec.shape[1]==len(sig) // (sample_rate//1000 * 16) + 1
    True
    >>> type(spec)
    <class 'numpy.ndarray'>
    >>> spec.dtype
    dtype('complex64')

    :param signal:
    :param sr:
    :return:
    """
    N = (fft_len * sr) // 1000
    return librosa.stft(signal, n_fft=N, hop_length=N // 2)


def get_common_space(W1, W2, threshold=common_space_err):
    """

    Example:
    >>> W1 = np.mat([[0.5, 0, 0], [0.5, 1, 0], [0, 0, 0], [0, 0, 1]])
    >>> W2 = np.mat([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    >>> get_common_space(W1, W2)
    matrix([[1, 0],
            [0, 1],
            [0, 0],
            [0, 0]])

    :param W1:
    :param W2:
    :return:
    """
    common_base = []
    H, _ = nmf_with_W(W2, W1)
    X = np.dot(W1, H)
    for i in range(X.shape[1]):
        tmp = X[:, i]
        err = np.mean(np.power(tmp-W2[:, i], 2))
        # print(err)
        if err < threshold:
            common_base.append(W2[:, i])
    if common_base:
        return np.column_stack(common_base)
    else:
        return None


def remove_common_space(W_list, W_cs, threshold=common_space_err):
    """
    remove basis in W_list which can be expressed by W_cs

    Example:
    >>> W1 = np.mat([[0.5, 0, 0], [0.5, 1, 0], [0, 0, 0], [0, 0, 1]])
    >>> W2 = np.mat([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    >>> Ws = [W1, W2]
    >>> W_cs = np.mat([[1, 0],[0, 1],[0, 0],[0, 0]])
    >>> remove_common_space(Ws, W_cs)
    [matrix([[ 0.],
            [ 0.],
            [ 0.],
            [ 1.]]), matrix([[0],
            [0],
            [1],
            [0]])]

    :param W_list:
    :param W_cs:
    :return:
    """
    if W_cs is None:
        return W_list
    res = []
    for w in W_list:
        tmp_arr = []
        H, _ = nmf_with_W(w, W_cs)
        X = np.dot(W_cs, H)
        for i in range(X.shape[1]):
            tmp = X[:, i]
            err = np.mean(np.power(tmp - w[:, i], 2))
            # print(err)
            if err > threshold:
                tmp_arr.append(w[:, i])
        try:
            res.append(np.column_stack(tmp_arr))
        except Exception as e:
            res.append(None)
    return res
