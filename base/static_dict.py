# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:35 AM
# @Author  : Liu jiandong
# @FileName: static_dict.py
# @Blog    ：http://meepoljd.com

import os
import librosa
import numpy as np

import config
import nmf


class StaticDict:

    def __init__(self, noise_path, speech_path, rank=config.rank):
        self._rank = rank
        self._noise_dict = None
        self._speech_dict = None
        self._total_dict = None
        self._n_fft = 0
        self._win = ''
        self._hop_len = 0
        self._noise_path = noise_path
        self._speech_path = speech_path

    @property
    def rank(self):
        return self._rank

    @property
    def total_dict(self):
        return self._total_dict

    @property
    def noise_dict(self):
        return self._noise_dict

    @property
    def speech_dict(self):
        return self._speech_dict

    @property
    def n_fft(self):
        return self._n_fft

    @n_fft.setter
    def n_fft(self, value):
        self._n_fft = value

    @property
    def hop_len(self):
        return self._hop_len

    @hop_len.setter
    def hop_len(self, value):
        self._hop_len = value

    @property
    def win(self):
        return self._win

    @win.setter
    def win(self, value):
        self._win = value

    def _gen_spec(self, path):
        _spec = None
        _list = os.listdir(path)
        for file in _list:
            file_name = os.path.join(path, file)
            signal, _ = librosa.load(file_name, sr=config.sample_rate)
            spec = librosa.stft(signal, n_fft=self._n_fft, hop_length=self._hop_len, window=self._win)
            abs_spec = np.abs(spec)
            if _spec is None:
                _spec = abs_spec
            else:
                _spec = np.column_stack([_spec, abs_spec])
        return _spec

    def build_noise(self):
        _noise_spec = self._gen_spec(self._noise_path)
        self._noise_dict, _ = nmf.decompose(_noise_spec)

    def build_speech(self):
        _speech_spec = self._gen_spec(self._speech_path)
        self._speech_dict, _ = nmf.decompose(_speech_spec)

    def merge_dic(self):
        if self._noise_dict is not None and self._speech_dict is not None:
            self._total_dict = np.column_stack([self._speech_dict, self._noise_dict])

    def init_dic(self):
        self.build_noise()
        self.build_speech()
        self.merge_dic()
