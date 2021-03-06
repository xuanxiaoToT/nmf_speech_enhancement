# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: mono_stream.py
# @Blog    ：http://meepoljd.com


import wave
import pyaudio

from base import BaseClass


class MonoStream(BaseClass):

    def __init__(self):
        self._duration = 0
        self._sample_rate = 0
        self._sample_width = 0
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
        elif int(len(str_data) / 2.0) < 2*self._duration:
            for _ in range(2*self._duration - len(str_data)):
                str_data += b'\x00'
        return str_data
