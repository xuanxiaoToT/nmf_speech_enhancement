# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: stream_builder.py
# @Blog    ：http://meepoljd.com


from . import *


class StreamBuilder:

    @staticmethod
    def _parse_ptype(dtype):
        if dtype == np.int16:
            return pyaudio.paInt16

    def __init__(self, stream_type='record', filename='', sr=sample_rate, chunk_size=chunk_time, dtype=np.int16):
        self._stream_type = stream_type
        self._sr = sr
        self._filename = filename
        self._chunk_size = chunk_size
        self._dtype = dtype

        if self._stream_type == 'record':
            self._stream = RecordStream()
            self._stream.chunk_size = self._chunk_size
            p = pyaudio.PyAudio()
            new_stream = p.open(format=StreamBuilder._parse_ptype(self._dtype), channels=1, input=True,
                                frames_per_buffer=self._chunk_size)
            self._stream.stream = new_stream
        elif self._stream_type == 'file':
            self._stream = FileStream()
            self._stream.chunk_size = self._chunk_size
            if filename == '':
                raise Exception('Build a file stream without filename')
            wav = wave.open(filename, 'r')
            self._stream.stream = wav

    def get_instance(self):
        return self._stream
