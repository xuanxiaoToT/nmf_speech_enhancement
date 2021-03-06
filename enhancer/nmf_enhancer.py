# -*- coding: utf-8 -*-
# @Time    : 6/14/18 12:00 PM
# @Author  : Liu jiandong
# @FileName: nmf_enhancer.py
# @Blog    ：http://meepoljd.com

import numpy as np
from scipy.signal import get_window
import scipy.fftpack as fft

import nmf
from base import BaseClass


class NmfEnhancer(BaseClass):

    def __init__(self, stream, dic):
        self._stream = stream
        self._dict = dic
        self._dict.n_fft = stream.n_fft
        self._dict.hop = stream.hop
        self._dict.win = stream.win
        self._dict.init_dic()
        self._signal_buffer = np.zeros(self._stream.n_fft + self._stream.hop)
        self._window_buffer = np.zeros(self._stream.n_fft + self._stream.hop)
        self._window_arr = get_window(self._stream.win, self._stream.n_fft, fftbins=True)

    def __next__(self):
        try:
            _, spec = self._stream.__next__()
            spec = np.asarray(self._enhance(spec))[0]
            spec = np.concatenate((spec, spec[-2:0:-1].conj()), 0)
            real_part = fft.ifft(spec).real
            y_tmp = self._window_arr * real_part
            self._signal_buffer[:self._stream.n_fft] = self._signal_buffer[:self._stream.n_fft] + y_tmp
            self._window_buffer[:self._stream.n_fft] = self._window_buffer[:self._stream.n_fft] + 1
            res = self._signal_buffer[:self._stream.hop] / self._window_buffer[:self._stream.hop]
            self._signal_buffer = np.concatenate([self._signal_buffer[self._stream.hop:], np.zeros(self._stream.hop)])
            self._window_buffer = np.concatenate([self._window_buffer[self._stream.hop:], np.zeros(self._stream.hop)])
            return res
        except StopIteration:
            if np.all(self._signal_buffer == 0) and np.all(self._window_buffer == 0):
                raise StopIteration
            else:
                index = self._window_buffer != 0
                res = self._signal_buffer[index] / self._window_buffer[index]
                self._signal_buffer = np.zeros(self._stream.n_fft + self._stream.hop)
                self._window_buffer = np.zeros(self._stream.n_fft + self._stream.hop)
                return res

    def _enhance(self, spec):
        abs_spec = np.abs(np.mat(spec))
        act = nmf.decompose_with_dict(abs_spec, self._dict.total_dict)
        abs_res = abs_spec - 0.9 * np.dot(act[:, self._dict.rank:], self._dict.noise_dict)
        # abs_res = np.dot(act[:, :self._dict.rank], self._dict.speech_dict)
        mask = abs_res / abs_spec
        norm = np.sum(abs_spec) / np.sum(abs_res)
        return np.multiply(spec, mask) * norm
