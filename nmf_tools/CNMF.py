import numpy as np

def cmfwisa(V, rank, config = None):
    """
    复数NMF分解的工具
    :param V: 待分解的矩阵
    :param rank: 基的个数
    :param config: 一些设置
    :return:
    """
    m, n = V.shape()
    # 创建相位的矩阵的初始值
    if not hasattr(config, "P_init"):
        config.P_init = np.exp(1j * np.angle(V))
    # 迭代过程中是否更新相位矩阵P
    if not hasattr(config, "P_fixed"):
        config.P_fixed = False
    if not hasattr(config, "W_init"):
        config.W_init = np.random.random((m, rank))
    if not hasattr(config, "H_init"):
        config.H_init = np.random.random((rank, n))
    if not hasattr(config, "max_iter"):
        config.max_iter = 200
    W = np.copy(config.W_init)
    P = np.copy(config.P_init)
    H = np.copy(config.H_init)
    V_hat = np.dot(W, H) * P
    V_bar