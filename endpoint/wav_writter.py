# -*- coding: utf-8 -*-
# @Time    : 6/21/18 12:16 PM
# @Author  : Liu jiandong
# @FileName: wav_writter.py
# @Blog    ：http://meepoljd.com

import wave
import numpy as np

from base import BaseClass


class WavWriter(BaseClass):

    def __init__(self, stream, output):
        self._stream = stream
        self._sample_rate = stream.sample_rate
        self._output = output
        self._buffer = np.array([])

    def __next__(self):
        try:
            seg = self._stream.__next__()
            self._buffer = np.concatenate([self._buffer, seg])
            return seg
        except StopIteration:
            f = wave.open(self._output, 'wb')
            f.setnchannels(1)
            f.setsampwidth(self.sample_width)
            f.setframerate(self.sample_rate)
            f.writeframes(self._buffer.astype(self.width2dtype()).tostring())
            f.close()
            raise StopIteration
