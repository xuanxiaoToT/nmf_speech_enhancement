def nmf_enhance(mix_file, W_dic):
    """
    use standard NMF tool to enhance wav file(single channel)

    Example:
    >>> import scipy.io as sio
    >>> dic = sio.loadmat('mats/W.mat')
    >>> clean = nmf_enhance('mix/fjcs0_sx319.wav_0.wav', dic)
    >>> write_wav(clean, 'doc_test/test.wav')

    :param signal:
    :param W_signal:
    :param W_noise:
    :return:
    """
    W_signal = W_dic['w_speech']
    W_noise = W_dic['w_noise']
    signal, _ = librosa.load(mix_file, sr=sample_rate)
    if emphasis:
        signal = pre_emphasise(signal)
    mix_spec = get_spec(signal)
    mix_abs = np.abs(mix_spec)
    len_sig, len_noise = W_signal.shape[1], W_noise.shape[1]
    W = np.column_stack([W_signal, W_noise])
    H, _ = nmf_with_W(mix_abs, W)
    H_noise = H[len_sig:, :]
    noise_spec = np.dot(W_noise, H_noise)
    # Normal way
    # sig_abs = np.dot(W_signal, H[:len_sig, :])
    # A trick
    sig_abs = mix_abs - .9 * noise_spec
    signal = spec2sig(sig_abs, mix_file)
    if emphasis:
        signal = de_emphasise(signal)
    return signal
