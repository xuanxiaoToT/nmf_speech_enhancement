import os
import librosa
import scipy.io as sio
from matplotlib import pyplot
from pystoi.stoi import stoi
from sklearn.externals import joblib
from nmf.dsp import *


def clean_models():
    if os.path.exists(os.path.join(mat_path, 'W.mat')):
        os.remove(os.path.join(mat_path, 'W.mat'))
    if os.path.exists(os.path.join(mat_path, 'model.mat')):
        os.remove(os.path.join(mat_path, 'model.mat'))


def load_models():
    dic = sio.loadmat(os.path.join(mat_path, 'W.mat'))
    if 'w_cs' in dic.keys():
        mlp = joblib.load(os.path.join(mat_path, 'model.mat'))
        dic['model'] = mlp
    return dic


def plot_spec(signal, sr=sample_rate, title=''):
    """
    to polt a spec quickly, wrap it

    Example:
    >>> sig, _ = librosa.load('speech/test/fjcs0_sx409.wav', sr=sample_rate)
    >>> plot_spec(sig)

    >>> plot_spec(sig, sample_rate)

    >>> plot_spec(sig, sample_rate, 'test_title')

    :param signal:
    :param sr:
    :param title:
    :return:
    """
    pyplot.figure()
    pyplot.title(title)
    pyplot.specgram(signal, Fs=sr, scale_by_freq=True, sides='default')
    pyplot.show()





def spec2sig(spec, mix_file, sr=sample_rate):
    """
    use phase of mix spec to restruct the signal

    Example:
    >>> sig, _ = librosa.load('speech/test/fjcs0_sx319.wav', sr=sample_rate)
    >>> spec = get_spec(sig)
    >>> new_sig = spec2sig(spec, 'mix/fjcs0_sx319.wav_0.wav')
    >>> print(len(sig)==len(new_sig))
    True

    :param spec:
    :param mix_file:
    :param sr:
    :return:
    """
    file, _ = librosa.load(mix_file, sr=sr)
    N = (32 * sr) // 1000
    mix = librosa.stft(file, n_fft=N, hop_length=N // 2)
    mask = spec / np.abs(mix)
    # 这里使用了维纳滤波的合成公式
    signal_spec = np.array(mix) * np.array(mask)
    sig = librosa.istft(signal_spec, hop_length=N // 2)
    return sig


def write_wav(signal, output, sr=sample_rate):
    """
    wrapped librosa write_wav to use the same sample_rate
    resample may cause diff between ori_sig and sig in wav

    Example:
    >>> write_wav(np.ndarray([1, 2]), 'doc_test/test.wav')

    >>> print(os.path.exists('doc_test'))
    True

    >>> print(os.path.exists('doc_test/test.wav'))
    True

    >>> write_wav([1,2,3], 'doc_test/test_err.wav')
    Traceback (most recent call last):
    ...
    librosa.util.exceptions.ParameterError: data must be of type numpy.ndarray

    >>> sig, fs = librosa.load('speech/test/fjcs0_sx319.wav', sr=None)
    >>> write_wav(sig, 'doc_test/same_test.wav', sr=fs)
    >>> new_sig, _ = librosa.load('doc_test/same_test.wav', sr=fs)

    :param signal:
    :param output:
    :return:
    """
    path = os.path.dirname(output) # 进行路径的检查
    if not os.path.exists(path):
        os.mkdir(path)
    librosa.output.write_wav(output, signal, sr, norm=True)


def cal_pseq(enhance, clean, sr=sample_rate):
    """
    Use bin file to calculate PESQ of enhance speech

    Example:
    >>> cal_pseq('test', 'test')
    Traceback (most recent call last):
    ...
    Exception: you use the PESQ program with wrong command

    >>> cal_pseq('mix/fjcs0_sx319.wav_-1.wav', 'speech/test/fjcs0_sx319.wav')
    1.12

    :param enhance:
    :param clean:
    :return:
    """
    lines = os.popen("./tools/pesq +%s %s %s"%(sr, clean, enhance)).readlines()
    try:
        res = float(lines[-1].split('=')[-1])
    except ValueError:
        raise Exception('you use the PESQ program with wrong command')
    return res


def cal_stoi(enhance, clean, sr=sample_rate):
    """
    A wrapped pystoi func to get stoi

    Example:
    >>> cal_stoi('test', 'test')
    Traceback (most recent call last):
    ...
    Exception: can not find files, please check path

    >>> cal_stoi('mix/fjcs0_sx319.wav_-1.wav', 'speech/test/fjcs0_sx319.wav')
    0.76689639728190695

    :param enhance:
    :param clean:
    :return:
    """
    try:
        clean_signal, _ = librosa.load(clean, sr=sr)
        # plot_wav(clean_signal, title=clean)
        enhance_signal, _ = librosa.load(enhance, sr=sr)
        clean_signal = clean_signal[:len(enhance_signal)] # 有时信号不等长，所以这里对齐
    except FileNotFoundError:
        raise Exception('can not find files, please check path')
    except IsADirectoryError:
        raise Exception('can not find files, please check path')

    res = stoi(clean_signal, enhance_signal, sample_rate, extended=False)
    return res


def plot_box(arr, title='', range=[-.5, 4.5]):
    """
    plot a box

    Example:
    >>> plot_box([1,2,3,4,5,6,7], title='Title')
    >>> pyplot.close()

    >>> plot_box(123)
    Traceback (most recent call last):
    ...
    Exception: please use a data array to plot, but get <class 'int'>

    :param arr:
    :param title:
    :return:
    """
    if type(arr) not in [list, np.array]:
        raise Exception('please use a data array to plot, but get %s'%type(arr))
    pyplot.ylim(range)
    pyplot.boxplot(arr, labels=[title])
    pyplot.show()


def plot_wav(signal, sr=sample_rate, title=''):
    """
    plot waves of a wav file

    Example:
    >>> sig, sr = librosa.load('speech/test/fjcs0_sx319.wav')
    >>> plot_wav(sig)
    >>> pyplot.close()

    >>> plot_wav(sig, sr=sr)
    >>> pyplot.close()

    >>> plot_wav(sig, sr=sr, title='Title')
    >>> pyplot.close()

    >>> plot_wav(123)
    Traceback (most recent call last):
    ...
    Exception: please use a signal array to plot, but get <class 'int'>

    :param signal:
    :param sr:
    :param title:
    :return:
    """
    if type(signal) not in [list, np.array, np.ndarray]:
        raise Exception('please use a signal array to plot, but get %s'%type(signal))
    n_frames = len(signal)
    time = np.arange(0, n_frames) * (1.0 / sr)
    pyplot.figure()
    pyplot.title(title)
    pyplot.plot(time, signal, c="g")
    pyplot.xlabel("time (seconds)")
    pyplot.show()


def norm_W(W):
    """
    norm each col in W

    Example:
    >>> W = np.random.rand(5, 5)
    >>> norm_W(W)
    >>> np.sum(W, axis=0)
    array([ 1.,  1.,  1.,  1.,  1.])

    :param W:
    :return:
    """
    for i in range(W.shape[1]):
        div = np.sum(W[:, i])
        W[:, i] /= div


def get_stats(data):
    """

    :param data:
    :return:
    """
    mean = np.mean(data)
    std = np.std(data)
    return mean, std


if __name__ == '__main__':
    import doctest
    import shutil
    if os.path.exists('doc_test'):
        shutil.rmtree('doc_test')
    doctest.testmod(verbose=False)
    shutil.rmtree('doc_test')