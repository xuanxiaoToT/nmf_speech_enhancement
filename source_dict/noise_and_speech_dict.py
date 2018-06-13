from . import *


class n_s_dict:

    def __init__(self, K = rank):
        self._rank = K
        self._noise_dict = None
        self._speech_dict = None
        self._total_dict = None

    @property
    def rank(self):
        return self._rank

    @property
    def total_dict(self):
        return self._total_dict

    @property
    def noise_dict(self):
        return self._noise_dict

    @property
    def speech_dict(self):
        return self._speech_dict

    @staticmethod
    def _build_dict(path):
        _spec = None
        _list = os.listdir(path)
        for file in _list:
            file_name = os.path.join(path, file)
            signal, _ = librosa.load(file_name, sr=sample_rate)
            spec = np.abs(get_spec(signal))
            if _spec is None:
                _spec = spec
            else:
                _spec = np.column_stack([_spec, spec])
        return _spec

    def build_noise(self, n_path):
        self._noise_dict = self._build_dict(n_path)
        if self._speech_dict is not None:
            self._total_dict = np.column_stack([self._speech_dict, self._noise_dict])

    def build_speech(self, s_path):
        self._speech_dict = self._build_dict(s_path)
        if self._noise_dict is not None:
            self._total_dict = np.column_stack([self._speech_dict, self._noise_dict])
