from . import *


class StreamBuilder:

    @staticmethod
    def _parse_ptype(dtype):
        if dtype == np.int16:
            return pyaudio.paInt16

    def __init__(self, stream_type='record', filename='', sr=sample_rate, chunk_size=chunk_size, dtype=np.int16):
        self._stream_type = stream_type
        self._sr = sr
        self._filename = filename
        self._chunk_size = chunk_size

        self._stream = MonoStream()
        self._stream.chunk_size = self._chunk_size
        if self._stream_type == 'record':
            p = pyaudio.PyAudio()
            new_stream = p.open(format=StreamBuilder._parse_ptype(self._dtype), channels=1, input=True,
                                frames_per_buffer=self._chunk_size)
            self._stream.stream = new_stream
        elif self._stream_type == 'file':
            if filename == '':
                raise Exception('Build a file stream without filename')
            wav = wave.open(filename, 'r')
            self._stream.stream = wav

    def get_instance(self):
        return self._stream
