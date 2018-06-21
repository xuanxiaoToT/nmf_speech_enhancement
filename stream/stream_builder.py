# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: stream_builder.py
# @Blog    ：http://meepoljd.com


from . import *


class StreamBuilder:

    def __init__(self, stream_type, sr, duration, dtype=np.int16, filename=''):
        self._stream_type = stream_type
        self._sample_rate = sr
        self._filename = filename
        self._dtype = dtype

        if self._stream_type == 'record':
            self._stream = RecordStream()
            self._stream.sample_rate = self._sample_rate
            self._duration = int((duration / 1000) * self._sample_rate)
            self._stream.duration = self._duration
            p = pyaudio.PyAudio()
            new_stream = p.open(rate=self._sample_rate, format=self._parse_ptype(), channels=1, input=True,
                                frames_per_buffer=self._duration)
            self._stream.stream = new_stream
        elif self._stream_type == 'file':
            self._stream = FileStream()
            if filename == '':
                raise Exception('Build a file stream without filename')
            wav = wave.open(filename, 'r')
            self._stream.stream = wav
            self._sample_rate = wav.getframerate()
            self._sample_width = wav.getsampwidth()
            self._stream.sample_rate = self._sample_rate
            self._duration = int((duration / 1000) * self._sample_rate)
            self._stream.duration = self._duration

    def get_instance(self):
        return self._stream

    def _parse_ptype(self):
        if self._dtype == np.int16:
            return pyaudio.paInt16
