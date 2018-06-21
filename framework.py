# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: framework.py
# @Blog    ：http://meepoljd.com

import config
from stream import StreamBuilder
from vad import Vad
from fft import FFT
from dictionary import StaticDict
from enhancer import NmfEnhancer
from endpoint import WavWriter


def main():
    builder = StreamBuilder(stream_type='file', filename='data/speech/train/fadg0_sa1.wav', sr=-1, duration=30)
    file_stream = builder.get_instance()
    vad_stream = Vad(file_stream)
    fft_stream = FFT(vad_stream, 32, 16)
    dic = StaticDict('data/debug/noise', 'data/debug/speech', rank=config.rank)
    enhancer_stream = NmfEnhancer(fft_stream, dic)
    wav_writer = WavWriter(enhancer_stream, 'out.wav')
    for _ in wav_writer:
        pass


if __name__ == '__main__':
    main()
