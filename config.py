# 算法参数
sample_rate = 16000
rank = 50
tol = 0.1
eps = 1e-5
max_iter = 200000
fft_len = 32  # 单位ms
hop_rate = 0.5

# 这个参数的选择是在W归一化的基础上的，但随机性依然存在
common_space_err = 0.025
emphasis = False
emphasis_rate = 0.95

# 子空间算法参数
cs_mix_rate = 0.5
cs_mlp_layers = [5, 1000]

# 路径相关内容
noise_path = 'noise'
speech_path = 'speech/train'
test_path = 'speech/test'
mix_path = 'mix'
out_path = 'output'
mat_path = 'mats'

# stream setting
chunk_time = int((20 / 1000) * sample_rate)

# vad setting
vad_len = 20
