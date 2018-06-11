"""
get statistics of PESQ and STOI 
"""
from sklearn.neural_network import MLPRegressor
from evaluation.utils import *
from nmf_enhance import *


def cal_W(params):
    """
    用于计算字典与模型并保存
    :return:
    """
    #print('未找到模型文件，开始计算')
    mlp = MLPRegressor(max_iter=20000, verbose=False, tol=1e-32, early_stopping=False)
    noise_spec = None
    noise_list = os.listdir(noise_path)
    noise_list = filter(lambda x: 'factory' in x, noise_list)
    #print('开始加载拼接噪声文件')
    for n in noise_list:
        file_name = os.path.join(noise_path, n)
        signal, _ = librosa.load(file_name, sr=sample_rate)
        spec = np.abs(get_spec(signal))
        if noise_spec is None:
            noise_spec = spec
        else:
            noise_spec = np.column_stack([noise_spec, spec])
    #print('开始分解噪声谱')
    W_noise, H_noise, _ = nmf(noise_spec, k=params['K'], alpha=params['alpha'], l1_rate=params['l1_rate'])

    #print('开始加载训练语音')
    speech_spec = None
    speech_list = os.listdir(speech_path)
    for s in speech_list:
        file_name = os.path.join(speech_path, s)
        signal, _ = librosa.load(file_name, sr=sample_rate)

        spec = np.abs(get_spec(signal))
        if speech_spec is None:
            speech_spec = spec
        else:
            speech_spec = np.column_stack([speech_spec, spec])
    #print('开始分解语音谱')
    W_speech, H_speeh, _ = nmf(speech_spec, k=params['K'], alpha=params['alpha'], l1_rate=params['l1_rate'])

    #print('求解公共子空间')
    W_cs = get_common_space(W_speech, W_noise, threshold=params['cs_tol'])
    #print('解得子空间包含%d组基' % W_cs.shape[1])

    #print('计算各个声源特征子空间')
    Ws = remove_common_space([W_speech, W_noise], W_cs, threshold=params['cs_tol'])
    W_speech_cs = Ws[0]
    W_noise_cs = Ws[1]
    #print('保存字典文件')
    if W_cs is None:
        sio.savemat(os.path.join(mat_path, 'W.mat'), {
            "w_speech": W_speech,
            "w_noise": W_noise,
        })
    else:
        sio.savemat(os.path.join(mat_path, 'W.mat'), {
            "w_speech": W_speech,
            "w_noise": W_noise,
            "w_cs": W_cs,
            "w_speech_cs": W_speech_cs,
            "w_noise_cs": W_noise_cs
        })
        #print('开始训练MLP模型')
        W_tmp = np.column_stack([W_noise_cs, W_cs])
        H_tmp, _ = nmf_with_W(noise_spec, W_tmp)
        H_tmp = np.mat(H_tmp)
        H_input = np.transpose(H_tmp[:W_noise_cs.shape[1], :])
        H_output = np.transpose(H_tmp[W_noise_cs.shape[1]:, :])
        mlp.fit(H_input, H_output)
        del noise_spec
        del speech_spec
        #print('保存MLP模型')
        joblib.dump(mlp, os.path.join(mat_path, 'model.mat'))


def main(params, plot=False):
    # if not os.path.exists(os.path.join(mat_path, 'W.mat')):
    #     cal_W()
    # print('加载已存在模型')
    dic = load_models()
    # print('开始降噪测试')
    mix_list = os.listdir(mix_path)
    pesq_arr = []
    stoi_arr = []
    pesq_arr_rate = []
    stoi_arr_rate = []
    # pesq_arr_mlp = []
    # stoi_arr_mlp = []
    for m in mix_list:
        file_name = os.path.join(mix_path, m)
        ori_wav_path = os.path.join(test_path, m.split('_')[0] + '_' + m.split('_')[1] + '.wav')
        nmf_out_path = os.path.join(out_path, m)
        cnmf_rate_out_path = os.path.join(out_path, m.split('.')[0] + 'rate.wav')
        cnmf_mlp_out_path = os.path.join(out_path, m.split('.')[0] + 'mlp.wav')

        clean_signal = nmf_enhance(file_name, dic)
        write_wav(clean_signal, nmf_out_path)
        pesq_arr.append(cal_pseq(nmf_out_path, ori_wav_path))
        stoi_arr.append(cal_stoi(nmf_out_path, ori_wav_path))
        if 'w_cs' in dic.keys():
            clean_signal_rate = nmfcs_rate_enhance(file_name, dic, cs_rate=params['cs_rate'])

            write_wav(clean_signal_rate, cnmf_rate_out_path)

            pesq_arr_rate.append(cal_pseq(cnmf_rate_out_path, ori_wav_path))
            stoi_arr_rate.append(cal_stoi(cnmf_rate_out_path, ori_wav_path))
        #     if dic['w_cs'].shape[1] > 3:
        #         clean_signal_mlp = nmfcs_mlp_enhance(file_name, dic)
        #         write_wav(clean_signal_mlp, cnmf_mlp_out_path)
        #         pesq_arr_mlp.append(cal_pseq(cnmf_mlp_out_path, ori_wav_path))
        #         stoi_arr_mlp.append(cal_stoi(cnmf_mlp_out_path, ori_wav_path))
    print('---------------Params---------------')
    print(params)
    print('---------------NMF------------------')
    avg_pesq, std_pesq = get_stats(pesq_arr)
    print("pesq %.2f +- %.2f" % (avg_pesq, std_pesq))
    avg_stoi, std_stoi = get_stats(stoi_arr)
    print("stoi %.2f +- %.2f" % (avg_stoi, std_stoi))
    if 'w_speech_cs' in dic.keys():
        print('--------------CNMF-basis------------')
        print(dic['w_cs'].shape[1])
        print(dic['w_speech_cs'].shape[1])
        print(dic['w_noise_cs'].shape[1])
        print('--------------CNMF-RATE-------------')
        avg_pesq, std_pesq = get_stats(pesq_arr_rate)
        print("pesq %.2f +- %.2f" % (avg_pesq, std_pesq))
        avg_stoi, std_stoi = get_stats(stoi_arr_rate)
        print("stoi %.2f +- %.2f" % (avg_stoi, std_stoi))
    #     if dic['w_cs'].shape[1] > 3:
    #         print('--------------CNMF-MLP--------------')
    #         avg_pesq, std_pesq = get_stats(pesq_arr_mlp)
    #         print("pesq %.2f +- %.2f" % (avg_pesq, std_pesq))
    #         avg_stoi, std_stoi = get_stats(stoi_arr_mlp)
    #         print("stoi %.2f +- %.2f" % (avg_stoi, std_stoi))
    print('------------------------------------')
    os.remove('_pesq_itu_results.txt')
    os.remove('_pesq_results.txt')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    params = {}
    for _ in range(10):
        params['alpha'] = 0.8
        params['l1_rate'] = 1
        params['cs_rate'] = 1.0
        params['cs_tol'] = 0.05
        params['K'] = 100
        clean_models()
        cal_W(params)
        main(params)
