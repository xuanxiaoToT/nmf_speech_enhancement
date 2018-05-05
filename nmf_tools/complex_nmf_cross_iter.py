"""
SUMMARY:  Non negative matrix (NMF)
          With Euclidean distance, KL divergence, beta divergence
AUTHOR:   Qiuqiang Kong
          q.kong@surrey.ac.uk
Liscence: BSD
Reference:https://github.com/audiofilter/nmflib
Created:  2016.06.14
Modified: 2017.11.05
--------------------------------------
Params:   V         F*N matrix, F, N is number of features, samples
          n_basis   rank of decomposition
          n_iter    number of iteration
          norm_W    type of normalization on W, can be 0 (None), 1 (l1-norm), 2 (l2-norm)
          norm_H    type of normalization on H, can be 0 (None), 1 (l1-norm), 2 (l2-norm)
          W0        initial value of W, (default: random)
          H0        initial vlaue of H, (default: random)
          update_W  whether update W, (default: True)
          update_H  whether update H, (default: True)
return:   W         learned dictionary, size: F*K, where F, K is number of features, rank of decomposition
          H         learned representation, size: K*N, where K, N is rank of decomposition, number of samples
--------------------------------------
"""
import numpy as np


# normalize matrix by row, norm can be 1 or 2
def normalize(X, norm_X):
    if norm_X == 1:
        X /= np.sum(np.abs(X), axis=0)
    if norm_X == 2:
        X /= np.sqrt(np.sum(X * X, axis=0))
    return X


def clip(X, eps):
    return np.clip(X, eps, np.inf)


def cnmf_euc(V, n_basis, n_iter=10, norm_W=0, norm_H=0, W0=None, H0=None, update_W=True, update_H=True, verbose=0):
    eps = 1e-8
    (F, N) = V.shape
    K = n_basis
    Vr = np.real(V)
    Vi = np.imag(V)
    if W0 is None:
        Wr = 0.1 * np.random.rand(F, K)
        Wi = 0.1 * np.random.rand(F, K)
    else:
        Wr = np.real(W0)
        Wi = np.imag(W0)
    if H0 is None:
        H = 0.1 * np.random.rand(K, N)
    else:
        H = H0

    for n in range(n_iter):
        # np.max(np.dot(np.dot(W.T, W), H), eps)
        if verbose:
            print('epoch', n)
        if update_W is True:
            Wr = Wr * np.dot(Vr, H.T) / clip(np.dot(np.dot(Wr, H), H.T), eps)
        if update_H is True:
            H = H * np.dot(Wr.T, Vr) / clip(np.dot(np.dot(Wr.T, Wr), H), eps)
        if update_W is True:
            Wi = Wi * np.dot(Vi, H.T) / clip(np.dot(np.dot(Wi, H), H.T), eps)
        if update_H is True:
            H = H * np.dot(Wi.T, Vi) / clip(np.dot(np.dot(Wi.T, Wi), H), eps)

        if n % 100 == 0:
            err = np.sum((np.abs((np.dot(Wr + 1j * Wi, H) - V)))  ) / (F*N)
            print("iter %d: \t %.10f"% (n, err))

        if norm_W != 0:
            Wr = normalize(Wr, norm_W)
            Wi = normalize(Wi, norm_W)
        if norm_H != 0:
            H = normalize(H, norm_H)

    return Wr + (1j * Wi), H, np.dot(Wr + (1j * Wi), H)
