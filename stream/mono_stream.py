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
        str_data = self._stream.read(self._chunk_size)
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
        str_data = self._stream.readframes(self._chunk_size)
        if str_data == b'':
            raise StopIteration()
        return str_data
