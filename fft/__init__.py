# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: __init__.py
# @Blog    ：http://meepoljd.com

import numpy as np
from scipy.signal import get_window

from config import sample_rate, fft_len, hop_rate
from .FFT import FFT
