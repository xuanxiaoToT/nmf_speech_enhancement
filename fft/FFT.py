# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: FFT.py
# @Blog    ：http://meepoljd.com

from . import *


class FFT:

    def __init__(self, stream, sr=sample_rate, win_len=fft_len, hop=hop_rate, window='hann'):
        self._sample_rate = sample_rate
        self._win_len = win_len
        self._fft_count = sr * (win_len / 1000)
        self._hop = hop
        self._hop_count = int(self._fft_count)
        self._signal_arr = np.array([])
        self._vad_arr = np.array([])
        self._stream = stream
        self._win = window
        self._win_array = get_window(self._win, self._win_len, fftbins=False)

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, value):
        self._stream = value

    @property
    def hop_count(self):
        return self._hop_count

    @property
    def win(self):
        return self._win

    @property
    def fft_count(self):
        return self._fft_count

    def __next__(self):
        try:
            while len(self._signal_arr) < self._fft_count:
                vad_flag, new_sig = self._stream.__next__()
                self._signal_arr = np.concatenate([self._signal_arr, new_sig])
                if vad_flag:
                    self._vad_arr = np.concatenate([self._vad_arr, np.ones(len(new_sig))])
                else:
                    self._vad_arr = np.concatenate([self._vad_arr, np.zeros(len(new_sig))])
        except StopIteration:
            raise StopIteration
        win_sig = self._win_array * self._signal_arr[:self._win_len]
        res = np.fft.fft(win_sig)
        res_flag = (np.sum(self._vad_arr[:self._win_len]) / self._win_len) > 0.5
        self._signal_arr = self._signal_arr[self._win_len:]
        self._vad_arr = self._vad_arr[self._win_len:]
        return res_flag, res

    def __iter__(self):
        return self
