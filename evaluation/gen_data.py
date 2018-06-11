from evaluation.utils import *


def mix_noise(signal, noise_wav, snr):
    """
    对信号与噪声按指定的SNR进行混合

    Example:

    >>> import librosa
    >>> sig, _ = librosa.load('speech/test/fjcs0_sx319.wav', sr=sample_rate)
    >>> noise, _ = librosa.load('noise/f16.wav', sr=sample_rate)
    >>> mix = mix_noise(sig, noise, 0)

    >>> print(len(mix)==len(sig))
    True

    :param signal:纯净信号
    :param noise_wav:噪声信号
    :param snr:信噪比
    :return:混合信号
    """
    # 对齐噪声
    num_padd = len(signal) // len(noise_wav) + 1
    noise = np.repeat(noise_wav, num_padd)
    cut_end = len(noise) - len(signal)
    # 随机切割点
    cut = np.random.randint(0, cut_end+1)
    noise = np.array(noise[cut:cut+len(signal)])
    # 计算噪声调整比例
    e_noise = np.sum(noise ** 2)
    e_signal = np.sum(signal ** 2)
    x = e_signal / ((10 ** snr) * e_noise)
    return noise * x + signal


def gen_mix_folder(speech_path, noise_path, snr_list, output_path):
    """
    根据规划好的流程，对测试噪声文件夹与测试语音中的wav文件以指定的snr进行混合并保存wav文件。

    Example:
    >>> gen_mix_folder('speech/test', 'noise', [-2, 0, 2], 'doc_test')
    mix fjem0_sx94.wav with snr = -1
    mix fjem0_sx94.wav with snr = 0
    mix fjem0_sx94.wav with snr = 1
    mix fjcs0_sx409.wav with snr = -1
    mix fjcs0_sx409.wav with snr = 0
    mix fjcs0_sx409.wav with snr = 1
    mix fjem0_sx4.wav with snr = -1
    mix fjem0_sx4.wav with snr = 0
    mix fjem0_sx4.wav with snr = 1
    mix fjem0_si1264.wav with snr = -1
    mix fjem0_si1264.wav with snr = 0
    mix fjem0_si1264.wav with snr = 1
    mix fjem0_sx274.wav with snr = -1
    mix fjem0_sx274.wav with snr = 0
    mix fjem0_sx274.wav with snr = 1
    mix fjem0_si634.wav with snr = -1
    mix fjem0_si634.wav with snr = 0
    mix fjem0_si634.wav with snr = 1
    mix fjem0_sa1.wav with snr = -1
    mix fjem0_sa1.wav with snr = 0
    mix fjem0_sa1.wav with snr = 1
    mix fjem0_si1894.wav with snr = -1
    mix fjem0_si1894.wav with snr = 0
    mix fjem0_si1894.wav with snr = 1
    mix fjem0_sx184.wav with snr = -1
    mix fjem0_sx184.wav with snr = 0
    mix fjem0_sx184.wav with snr = 1
    mix fjem0_sa2.wav with snr = -1
    mix fjem0_sa2.wav with snr = 0
    mix fjem0_sa2.wav with snr = 1
    mix fjem0_sx364.wav with snr = -1
    mix fjem0_sx364.wav with snr = 0
    mix fjem0_sx364.wav with snr = 1
    mix fjcs0_sx319.wav with snr = -1
    mix fjcs0_sx319.wav with snr = 0
    mix fjcs0_sx319.wav with snr = 1

    :param speech_path:
    :param noise_path:
    :param snr_list:
    :param output_path:
    :return:
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    speech_list = os.listdir(speech_path)
    noise_list = os.listdir(noise_path)
    noise_list = list(filter(lambda x: 'factory' in x, noise_list))
    for s in speech_list:
        signal, _ = librosa.load(os.path.join(speech_path,s), sr=sample_rate)
        noise_name = np.random.choice(noise_list, 1)[0]
        noise, _ = librosa.load(os.path.join(noise_path, noise_name), sr=sample_rate)
        for snr in snr_list:
            print('mix %s with snr = %d' % (s, snr))
            mix = mix_noise(signal, noise, snr)
            librosa.output.write_wav(os.path.join(output_path, "%s_%d.wav" % (s.split('.')[0], snr)), mix, sr=sample_rate)


if __name__ == '__main__':
    """
    这里进行函数的测试
    """
    # import doctest
    # import shutil
    # if os.path.exists('doc_test'):
    #     shutil.rmtree('doc_test')
    # doctest.testmod(verbose=False)
    # shutil.rmtree('doc_test')
    gen_mix_folder('speech/test', 'noise', [-2, 0, 2], 'mix')
