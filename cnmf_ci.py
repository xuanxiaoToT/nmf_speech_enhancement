from config import Rank, Shift
from sklearn.decomposition import NMF
from DSP import *
from fileManager import find_files
from nmf_tools import cnmf_euc


def move2positive(mat, step=Shift):
    tmp = mat + step * (1 + 1j)
    return tmp

def move_back(mat, step=Shift):
    return mat - step * (1 + 1j)

def main():
    '''
    主函数，包含分离的主要过程
    :return:
    '''
    # 开始处理实部
    speaker = ["s1", "s2", ]
    spec_dic = {}

    for s in speaker:
        files = find_files(s)
        print("完成%s相关文件读取" % (s))
        for f in files:
            print(f)
            spec = getSpec(f)
            spec_dic[f] = spec

    V = merge(speaker, spec_dic)
    del (spec_dic)
    W = []
    H = []

    models_real = []
    models_imag = []

    for i, v in enumerate(V):
        W_tmp, H_tmp, Vrec = cnmf_euc(move2positive(v), Rank, n_iter=20000)
        W.append(W_tmp)

    W_all = np.column_stack(W)

    # 加载混合谱
    mix_spec = getSpec("mix/mix.wav")
    _, resH, _ = cnmf_euc(move2positive(mix_spec), Rank*2, n_iter=20000, W0=W_all, update_W=False)
    s1_part = np.dot(W[0], resH[:Rank, :])
    s2_part = np.dot(W[1], resH[Rank:, :])

    s1_part = move_back(s1_part)
    s2_part = move_back(s2_part)

    sample_rate=16000
    N = (32 * sample_rate) // 1000
    sig = librosa.istft(np.asarray(s1_part), hop_length=N // 2)
    librosa.output.write_wav("s1_complex_ci.wav", sig, sample_rate)
    sig = librosa.istft(np.asarray(s2_part), hop_length=N // 2)
    librosa.output.write_wav("s2_complex_ci.wav", sig, sample_rate)

if __name__ == '__main__':
    main()
