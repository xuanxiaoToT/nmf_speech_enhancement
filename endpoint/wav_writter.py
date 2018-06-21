# -*- coding: utf-8 -*-
# @Time    : 6/21/18 12:16 PM
# @Author  : Liu jiandong
# @FileName: wav_writter.py
# @Blog    ：http://meepoljd.com

import wave
import numpy as np


class WavWriter:

    def __init__(self, stream, output):
        self._stream = stream
        self._sample_rate = stream.sample_rate
        self._dtype = stream.dtype
        self._output = output
        self._buffer = np.array([])

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

    def __iter__(self):
        return self

    def __next__(self):
        try:
            seg = self._stream.__next__()
            self._buffer = np.concatenate([self._buffer, seg])
            return seg
        except StopIteration:
            f = wave.open(self._output, 'wb')
            f.setnchannels(1)
            if self._dtype == np.int16:
                f.setsampwidth(2)
            f.setframerate(self._sample_rate)
            f.writeframes(self._buffer.astype(self._dtype).tostring())
            f.close()
            raise StopIteration
