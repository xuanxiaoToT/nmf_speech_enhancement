# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: Vad.py
# @Blog    ：http://meepoljd.com

from . import *


class Vad:

    def __init__(self, stream):
        self._sample_rate = stream.sample_rate
        self._vad = webrtcvad.Vad()
        self._vad.set_mode(2)
        self._stream = stream

    @property
    def sample_rate(self):
        if hasattr(self, '_sample_rate'):
            return self._sample_rate
        else:
            return self._stream.sample_rate

    @property
    def dtype(self):
        if hasattr(self, '_dtype'):
            return self._dtype
        else:
            return self._stream.dtype

    def __next__(self):
        str_data = self._stream.__next__()
        _cur_data = np.fromstring(str_data, dtype=self.dtype)
        _cur = self._vad.is_speech(str_data, self._sample_rate)
        return _cur, _cur_data

    def __iter__(self):
        return self

    def close(self):
        self._stream.close()
