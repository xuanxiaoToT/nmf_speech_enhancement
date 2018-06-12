"""
这是整个的降噪系统框架的实现，包含VAD，噪声字典更新机制。
VAD使用DNN版本与目标频段比值两种方法测试。
"""
from stream import StreamBuilder
from vad import Vad
from fft import FFT


def main():
    builder = StreamBuilder(stream_type='file', filename='data/speech/train/fadg0_sa1.wav')
    file_stream = builder.get_instance()
    vad_stream = Vad(file_stream)
    fft_stream = FFT(vad_stream)
    for res in fft_stream:
        print(res)


if __name__ == '__main__':
    main()
