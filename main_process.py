""" 
@author: zoutai
@file: mynmf.py 
@time: 2018/03/11 
@description: 
"""

import scipy.io as sio
from sklearn.decomposition import NMF
from sklearn.neural_network import MLPRegressor
from DSP import *
from fileManager import find_files


def main():
    '''
    主函数，包含分离的主要过程
    :return:
    '''
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=500)
    speaker = ["s1", "s2", ]  # 这里是说话人的数据文件夹名称
    spec_dic = {}
    for s in speaker:
        files = find_files(s)  # 自己写的一个便利文件的工具
        print("完成%s相关文件读取" % (s))
        for f in files:
            print(f)
            spec = getSpec(f)  # 获取谱的函数，实现见DSP文件（通过STFT）短时傅里叶变换
            print("生成%s幅度谱" % (f))
            spec_dic[f] = np.abs(spec)  # 返回帧t出处的频谱大小

    V = merge(speaker, spec_dic)  # 把这些谱拼成一个大的准备分解（是否拼接音频文件再转换成谱更好）
    del (spec_dic)  # 这里是去除掉之前一步的中间变量，如果数据量大，整个过程很费内存
    W = []
    H = []
    models = []
    print("清理内存")
    for i, v in enumerate(V):
        model = NMF(n_components=70, init='random', random_state=0)
        models.append(model)
        H.append(model.fit_transform(v.T))

        W.append(np.mat(model.components_))
        # 保存中间字典值
        print("保存说话人%d字典" % i)
        sio.savemat("myw" + str(i) + ".mat", {"w" + str(i + 1): np.mat(model.components_)})

    model = models[0]
    common_base = []
    for i in range(W[1].shape[0]):
        X = model.transform(W[1][i, :])
        err = np.sum(np.power(model.inverse_transform(X) - W[1][i, :], 2)) # 计算MSE误差
        if err < 1:
            common_base.append(W[1][i, :])
    W.append(np.row_stack(common_base))
    print("得到公共子空间元素数%d" % W[-1].shape[0])

    model.n_components_ = W[2].shape[0]
    model.components_ = W[2]
    for w in W[:-1]:
        count = 0
        res = []
        for i in range(w.shape[0]):
            X = model.transform(w[i, :])
            err = np.sum(np.power(model.inverse_transform(X) - w[i, :], 2))  # 计算MSE误差
            if err > 1:
                count += 1
                res.append(w[i, :])
        W.append(np.row_stack(res))
        print("计算得到的各说话人独立特征空间元素数：%d" % count)
    # 抛弃原始字典
    base_W1 = W.pop(0)
    base_W2 = W.pop(0)
    sio.savemat("r_common.mat", {"r_common": np.mat(W[0]).T})
    sio.savemat("r1.mat", {"r1": np.mat(W[1]).T})
    sio.savemat("r2.mat", {"r2": np.mat(W[2]).T})

    # 这里是基础算法的测试部分
    total_base_w = sum([base_W1.shape[0], base_W2.shape[0]])
    model.n_components = total_base_w
    model.n_components_ = total_base_w
    model.components_ = np.row_stack([base_W1, base_W2])
    # 加载混合谱
    mix_spec = getSpec("mix/mix.wav")
    mix_abs = np.abs(mix_spec)
    # 进行分解
    H = model.transform(mix_abs.T)
    s1_part = np.dot(H[:, :base_W1.shape[0]], base_W1).T
    s2_part = np.dot(H[:, base_W1.shape[0]:], base_W2).T
    s1_mask = s1_part / mix_abs
    s2_mask = s2_part / mix_abs

    reconstruct("mix/mix.wav", s1_mask, "./s1_sep.wav")
    reconstruct("mix/mix.wav", s2_mask, "./s2_sep.wav")

    # 准备一个数组用于将来的分解
    shape_count = [w.shape[0] for w in W]
    total_w = sum([w.shape[0] for w in W])

    print("拼接形成%d组基的字典" % total_w)
    model.n_components = total_w
    model.n_components_ = total_w
    model.components_ = np.row_stack(W)
    print("加载混合谱")
    mix_spec = getSpec("mix/mix.wav")
    print("转换为幅度谱")
    mix_abs = np.abs(mix_spec)
    print("分解得到激活系数")
    H = model.transform(mix_abs.T)

    common_part = np.dot(H[:, :shape_count[0]], W[0]).T
    common_part /= 2
    s1_part = np.dot(H[:, shape_count[0]:shape_count[1] + shape_count[0]], W[1]).T
    s2_part = np.dot(H[:, shape_count[1]+shape_count[0]:], W[2]).T

    abs_s1 = common_part + s1_part
    abs_s2 = common_part + s2_part

    s1_mask = abs_s1 / mix_abs
    s2_mask = abs_s2 / mix_abs

    reconstruct("mix/mix.wav", s1_mask, "./s1_cs.wav")
    reconstruct("mix/mix.wav", s2_mask, "./s2_cs.wav")

if __name__ == '__main__':
    main()
