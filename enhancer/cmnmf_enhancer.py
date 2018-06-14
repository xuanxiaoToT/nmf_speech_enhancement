def nmfcs_mlp_enhance(mix_file, W_dic):
    """

    :param mix_file:
    :param W_dic:
    :return:
    """
    W_signal = W_dic['w_speech_cs']
    W_noise = W_dic['w_noise_cs']
    W_cs = W_dic['w_cs']
    model = W_dic['model']
    signal, _ = librosa.load(mix_file, sr=sample_rate)
    mix_spec = get_spec(signal)
    mix_abs = np.abs(mix_spec)
    len_sig, len_noise, len_cs = W_signal.shape[1], W_noise.shape[1], W_cs.shape[1]
    W = np.column_stack([W_signal, W_noise, W_cs])
    H, _ = nmf_with_W(mix_spec, W)
    H_sig = H[:len_sig, :]
    H_noise = H[len_sig:len_sig+len_noise, :]
    H_cs = (np.mat(model.predict(H_noise.T))).T
    cs_spec = np.dot(W_cs, H_cs)
    # sig_spec = np.dot(W_signal, H_sig)
    noise_spec = np.dot(W_noise, H_noise)
    sig_spec = mix_abs -  noise_spec - cs_spec
    signal = spec2sig(sig_spec, mix_file)
    return signal