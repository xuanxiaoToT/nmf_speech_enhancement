# -*- coding: utf-8 -*-
# @Time    : 6/21/18 7:24 PM
# @Author  : Liu jiandong
# @FileName: base_class.py
# @Blog    ：http://meepoljd.com
import numpy as np


class BaseClass:

    @property
    def sample_rate(self):
        if hasattr(self, '_sample_rate'):
            return self._sample_rate
        elif hasattr(self, '_stream'):
            return self._stream.sample_rate
        else:
            raise Exception('has no attr named sample_rate')

    @sample_rate.setter
    def sample_rate(self, value):
        if hasattr(self, '_sample_rate'):
            self._sample_rate = value
        elif hasattr(self, '_stream'):
            self._stream._sample_rate = value
        else:
            raise Exception('has no attr named sample_rate')

    @property
    def sample_width(self):
        if hasattr(self, '_sample_width'):
            return self._sample_width
        elif hasattr(self, '_stream'):
            return self._stream.sample_width
        else:
            raise Exception('has no attr named sample_width')

    @sample_width.setter
    def sample_width(self, value):
        if hasattr(self, '_sample_rate'):
            self._sample_width = value
        elif hasattr(self, '_stream'):
            self._stream._sample_width = value
        else:
            raise Exception('has no attr named sample_width')

    @property
    def hop(self):
        if hasattr(self, '_hop'):
            return self._hop
        elif hasattr(self, '_stream'):
            return self._stream.hop
        else:
            raise Exception('has no attr named hop')

    @property
    def win(self):
        if hasattr(self, '_win'):
            return self._win
        elif hasattr(self, '_stream'):
            return self._stream.win
        else:
            raise Exception('has no attr named win')

    @property
    def n_fft(self):
        if hasattr(self, '_n_fft'):
            return self._n_fft
        elif hasattr(self, '_stream'):
            return self._stream.n_fft
        else:
            raise Exception('has no attr named n_fft')

    def __iter__(self):
        return self

    def width2dtype(self):
        i = self.sample_width * 8
        return getattr(np, 'int%d' % i)
