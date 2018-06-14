# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: Vad.py
# @Blog    ：http://meepoljd.com

from . import *


class Vad:

    def __init__(self, stream, dtype=np.int16, sr=sample_rate, v_len=vad_len,):
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise Exception('only use stream sampled at 8000, 16000, 32000 or 48000 Hz')
        if v_len not in [10, 20, 30]:
            raise Exception('A frame must be either 10, 20, or 30 ms in duration')
        self._sample_rate = sr
        self._frame_len = v_len
        self._arr_len = self._sample_rate * (self._frame_len / 1000)
        self._vad = webrtcvad.Vad()
        self._vad.set_mode(2)
        self._stream = stream
        self._dtype = dtype

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        if value not in [8000, 16000, 32000, 48000]:
            raise Exception('only use stream sampled at 8000, 16000, 32000 or 48000 Hz')
        self._sample_rate = value

    @property
    def frame_len(self):
        return self._frame_len

    @frame_len.setter
    def frame_len(self, value):
        if value not in [10, 20, 30]:
            raise Exception('A frame must be either 10, 20, or 30 ms in duration')
        self._frame_len = value

    @property
    def hop(self):
        return self._hop

    @hop.setter
    def hop(self, value):
        if not 0 <= value <= 1:
            raise Exception('hop must in [0,1]')
        self._hop = value

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    def __next__(self):
        str_data = self._stream.__next__()
        _cur_data = np.fromstring(str_data, dtype=self._dtype)
        if len(_cur_data) != self._arr_len:
            raise StopIteration
        _cur = self._vad.is_speech(str_data, self._sample_rate)
        return _cur, _cur_data

    def __iter__(self):
        return self

    def close(self):
        self._stream.close()
