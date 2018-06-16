# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: mono_stream.py
# @Blog    ：http://meepoljd.com


from . import *


class MonoStream:

    def __init__(self):
        self._duration = 0
        self._dtype = np.int16
        self._sample_rate = 0
        self._stream = None

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        self._duration = value

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    def close(self):
        try:
            self._stream.close()
        except Exception:
            raise Exception('some thing wrong when closing stream')

    def __iter__(self):
        return self


class RecordStream(MonoStream):

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, value):
        if type(value) is not pyaudio.PyAudio:
            raise Exception('Must put a pyAudio')
        self._stream = value

    def __next__(self):
        str_data = self._stream.read(self._duration)
        return str_data


class FileStream(MonoStream):

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, value):
        if type(value) is not wave.Wave_read:
            raise Exception('Must put a wave')
        self._stream = value

    def __next__(self):
        str_data = self._stream.readframes(self._duration)
        if str_data == b'':
            raise StopIteration()
        elif int(len(str_data) / 2.0) < int(len(str_data) / 2.0):
            str_data = str_data + [b'\x00\x00' for _ in range(int(self.sample_rate*(self.duration/1000))
                                                              - int(len(str_data) / 2.0))]
        return str_data
