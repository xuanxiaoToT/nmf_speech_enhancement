# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: FFT.py
# @Blog    ：http://meepoljd.com


import numpy as np
import scipy.signal as signal
import scipy.fftpack as fft
from base import BaseClass


class FFT(BaseClass):

    def __init__(self, stream, n_fft_ms, hop_ms, window='hann'):
        self._sample_rate = stream.sample_rate
        self._n_fft = int((n_fft_ms / 1000) * self._sample_rate)
        self._hop = int((hop_ms / 1000) * self._sample_rate)
        self._signal_arr = np.array([])
        self._vad_arr = np.array([])
        self._stream = stream
        self._win = window
        self._win_array = signal.get_window(self._win, self._n_fft, fftbins=False)

    def __next__(self):
        while len(self._signal_arr) < self._n_fft:
            vad_flag, new_sig = self._stream.__next__()
            self._signal_arr = np.concatenate([self._signal_arr, new_sig])
            if vad_flag:
                self._vad_arr = np.concatenate([self._vad_arr, np.ones(len(new_sig))])
            else:
                self._vad_arr = np.concatenate([self._vad_arr, np.zeros(len(new_sig))])
        win_sig = self._win_array * self._signal_arr[:self._n_fft]
        res = fft.fft(win_sig)
        res_flag = (np.mean(self._vad_arr[:self._n_fft])) > 0.5
        self._signal_arr = self._signal_arr[self._n_fft:]
        self._vad_arr = self._vad_arr[self._n_fft:]
        return res_flag, res[:(int(1 + self._n_fft // 2))]

    def __iter__(self):
        return self
