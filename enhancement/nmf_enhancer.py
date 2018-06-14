# -*- coding: utf-8 -*-
# @Time    : 6/14/18 12:00 PM
# @Author  : Liu jiandong
# @FileName: nmf_enhancer.py
# @Blog    ：http://meepoljd.com

from . import *


class NmfEnhancer:

    def __init__(self, stream, dic):
        self._stream = stream
        self._hop_count = self._stream.hop_count
        self._n_fft = self._stream.fft_count
        self._dict = dic
        self._spec = np.column_stack([self._stream.__next__()])
        self._signal_buffer = np.zeros(self._n_fft + self._hop_count)
        self._window_buffer = np.zeros(self._n_fft + self._hop_count)
        self._window_arr = get_window(self._stream.win, self._n_fft, fftbins=True)

    def __iter__(self):
        return self

    def __next__(self):
        spec = self._stream.__next__()
        spec = self._enhance(spec)
        spec = np.concatenate((spec, spec[-2:0:-1].conj()), 0)
        # spec = spec.flatten()
        y_tmp = self._window_arr * np.fft.ifft(spec).real
        self._signal_buffer[:self._n_fft] = self._signal_buffer[:self._n_fft] + y_tmp
        self._window_buffer[:self._n_fft] = self._window_buffer[:self._n_fft] + 1
        res = self._signal_buffer[:self._hop_count] / self._window_buffer[:self._hop_count]
        self._signal_buffer = np.concatenate([self._signal_buffer[self._hop_count:], np.zeros(self._hop_count)])
        self._window_buffer = np.concatenate([self._window_buffer[self._hop_count:], np.zeros(self._hop_count)])
        return res

    def _enhance(self, spec):
        abs_spec = np.abs(spec)
        H = nmf_with_W(np.mat(abs_spec).T, self._dict.total_dict)
        abs_res = spec - 0.9 * np.dot(self._dict.noise_dict, H[self._dict.rank:])
        return spec * (abs_res / abs_spec)
