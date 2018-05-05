import scipy.io as sio
from config import Rank, Shift
from sklearn.decomposition import NMF
from DSP import *
from fileManager import find_files

def sync_base(w_real, w_imag, h_real, h_imag):
    '''
    这个函数用于选择基的虚部，用于对齐两个部分的激活系数
    :param w_real:
    :param w_imag:
    :param h_real:
    :param h_imag:
    :return:
    '''
    # 假设已经进行了参数的校验
    for i in range(len(w_real)):
        Wr = w_real[i]
        Wi = w_imag[i]
        Hr = h_real[i]
        Hi = h_imag[i]
        model = NMF(n_components=Rank, init='random', random_state=0)
        model.fit(Hr.T)
        model.components_ = Hi.T
        Pt = model.transform(Hr.T)
        P = np.mat(Pt.T)
        err = np.dot(Hi, P) - Hr
        print(np.sum(np.abs(err)))
        Wip = np.dot(P.I, Wi)


def move2positive(mat, step=Shift):
    tmp =  mat + step
    print("经过平移原矩阵的最小值变为%d"%(np.min(tmp)))
    if np.min(tmp) <0 and np.min(tmp) > -5:
        tmp[tmp < 0] = 0
        print("进行二次规整")
    return tmp

def move_back(mat, step=Shift):
    if mat.dtype == np.complex:
        return mat - step * (1 + 1j)
    else:
        return mat - step

def main():
    '''
    主函数，包含分离的主要过程
    :return:
    '''
    # 开始处理实部
    speaker = ["s1", "s2", ]
    spec_dic_real = {}
    spec_dic_imag = {}
    for s in speaker:
        files = find_files(s)
        print("完成%s相关文件读取" % (s))
        for f in files:
            print(f)
            spec = getSpec(f)
            spec_dic_real[f] = np.real(spec)
            spec_dic_imag[f] = np.imag(spec)

    Vr = merge(speaker, spec_dic_real)  # 把这些谱拼成一个大的准备分解（是否拼接音频文件再转换成谱更好）
    Vi = merge(speaker, spec_dic_imag)
    del (spec_dic_real)  # 这里是去除掉之前一步的中间变量，如果数据量大，整个过程很费内存
    del (spec_dic_imag)
    Wr = []
    Hr = []
    Wi = []
    Hi = []

    models_real = []
    models_imag = []

    for i, v in enumerate(Vr):
        model = NMF(n_components=Rank, init='random', random_state=0)
        models_real.append(model)
        Hr.append(model.fit_transform(move2positive(v.T)))
        Wr.append(np.mat(model.components_))

    for i, v in enumerate(Vi):
        model = NMF(n_components=Rank, init='random', random_state=0)
        models_imag.append(model)
        Hi.append(model.fit_transform(move2positive(v.T)))
        Wi.append(np.mat(model.components_))

    # 规整复数
    sync_base(Wr, Wi, Hr, Hi)
    return
    # 这里是基础算法的测试部分
    m_real = models_real[0]
    m_imag = models_imag[0]

    total_base_w = Rank + Rank

    m_imag.n_components = total_base_w
    m_imag.n_components_ = total_base_w
    m_imag.components_ = np.row_stack(Wi)

    m_real.n_components = total_base_w
    m_real.n_components_ = total_base_w
    m_real.components_ = np.row_stack(Wr)

    # 加载混合谱
    mix_spec = getSpec("mix/mix.wav")
    mix_real = move2positive(np.real(mix_spec))
    mix_imag = move2positive(np.imag(mix_spec))

    H_real = m_real.transform(mix_real.T)
    H_imag = m_imag.transform(mix_imag.T)
    s1_part = np.dot(H_real[:, :Rank], Wr[0]).T + 1j * np.dot(H_imag[:, :Rank], Wi[0]).T
    s2_part = np.dot(H_real[:, Rank:], Wr[1]).T + 1j * np.dot(H_imag[:, Rank:], Wi[1]).T

    s1_part = move_back(s1_part)
    s2_part = move_back(s2_part)

    sample_rate=16000
    N = (32 * sample_rate) // 1000
    sig = librosa.istft(np.asarray(s1_part), hop_length=N // 2)
    librosa.output.write_wav("s1_complex.wav", sig, sample_rate)
    sig = librosa.istft(np.asarray(s2_part), hop_length=N // 2)
    librosa.output.write_wav("s2_complex.wav", sig, sample_rate)

if __name__ == '__main__':
    main()
