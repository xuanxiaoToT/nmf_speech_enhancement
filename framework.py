# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: framework.py
# @Blog    ：http://meepoljd.com

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
