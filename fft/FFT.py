from . import *


class FFT:

    def __init__(self, stream, sr=sample_rate, win_len=fft_len, hop=hop_rate, window=np.hanning):
        self._sample_rate = sample_rate
        self._win_len = win_len
        self._fft_count = sr * (win_len / 1000)
        self._hop = hop
        self._hop_count = int(self._fft_count)
        self._signal_arr = np.array([])
        self._vad_arr = np.array([])
        self._stream = stream
        self._win_func = window

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, value):
        self._stream = value

    def __next__(self):
        while len(self._signal_arr) < self._fft_count:
            vad_flag, new_sig = self._stream.__next__()
            np.concatenate([self._signal_arr, new_sig])
            if vad_len:
                np.concatenate([self._vad_arr, np.ones(len(new_sig))])
            else:
                np.concatenate([self._vad_arr, np.zeros(len(new_sig))])
        res = np.fft.fft(self._win_func(self._signal_arr[:self._win_len]))
        res_flag = (np.sum(self._vad_arr[:self._win_len]) / self._win_len) > 0.5
        self._signal_arr = self._signal_arr[self._win_len:]
        self._vad_arr = self._vad_arr[self._win_len:]
        return res_flag, res
