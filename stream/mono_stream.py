from . import *


class MonoStream:

    def __init__(self):
        self._chunk_size = 1024
        self._dtype = np.int16
        self._stream = None

    @property
    def chunk_size(self):
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        self._chunk_size = value

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, value):
        self._stream = value

    def __next__(self):
        if type(self._stream) is wave.Wave_read:
            str_data = self._stream.readframes(self._chunk_size)
        else:
            str_data = self._stream.read(self._chunk_size)
        return str_data

    def close(self):
        try:
            self._stream.close()
        except Exception:
            raise Exception('some thing wrong when closing stream')
