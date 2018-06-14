def nmfcs_rate_enhance(mix_file, W_dic, cs_rate=0.5):
    """

    :param mix_file:
    :param W_signal:
    :param W_noise:
    :return:
    """
    W_signal = W_dic['w_speech_cs']
    W_noise = W_dic['w_noise_cs']
    W_cs = W_dic['w_cs']
    signal, _ = librosa.load(mix_file, sr=sample_rate)
    mix_spec = get_spec(signal)
    mix_abs = np.abs(mix_spec)
    len_sig, len_noise, len_cs = W_signal.shape[1], W_noise.shape[1], W_cs.shape[1]
    W = np.column_stack([W_signal, W_noise, W_cs])
    H, _ = nmf_with_W(mix_abs, W)
    H_sig = H[:len_sig, :]
    H_noise = H[len_sig:len_sig+len_noise, :]
    H_cs = H[-len_cs:, :]
    cs_spec = np.dot(W_cs, H_cs)
    cs_spec *= cs_rate
    noise_spec = np.dot(W_noise, H_noise)
    sig_spec = np.dot(W_signal, H_sig)
    clean_spec = mix_abs - .9 * noise_spec - (1 - cs_rate) * cs_spec
    # clean_spec = sig_spec + cs_rate * cs_spec
    signal = spec2sig(clean_spec, mix_file)
    return signal
