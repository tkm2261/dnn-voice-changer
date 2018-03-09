# coding: utf-8

import numpy as np
from scipy.io import wavfile
import pyworld
import pysptk

from nnmnkwii import preprocessing as P

from hparams import vc as hp

from models import get_generator

FS = 16000
HOP_LENGHT = int(FS * (hp.frame_period * 0.001))


def generate_changed_voice(model, input_path):

    fs, x = wavfile.read(input_path)
    x = x.astype(np.float64)
    if len(x.shape) > 1:
        x = x.mean(axis=1)

    f0, timeaxis = pyworld.dio(x, fs, frame_period=hp.frame_period)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)
    alpha = pysptk.util.mcepalpha(fs)
    mc = pysptk.sp2mc(spectrogram, order=hp.order, alpha=alpha)
    c0, mc = mc[:, 0], mc[:, 1:]

    mc = P.modspec_smoothing(mc, FS / HOP_LENGHT, cutoff=50)
    mc = P.delta_features(mc, hp.windows).astype(np.float32)

    gen_data = model.predict(mc)

    gen_data = np.hstack([c0.reshape((-1, 1)), gen_data])

    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    spectrogram = pysptk.mc2sp(
        gen_data.astype(np.float64), alpha=alpha, fftlen=fftlen)
    waveform = pyworld.synthesize(
        f0, spectrogram, aperiodicity, fs, hp.frame_period)

    return waveform


if __name__ == '__main__':
    model_path = 'weights/generator_5800.hdf5'
    model = get_generator()
    model.load_weights(filepath=model_path)

    input_path = 'data/cmu_arctic/cmu_us_bdl_arctic/wav/arctic_a0079.wav'
    #input_path = 'test_input.wav'

    waveform = generate_changed_voice(model, input_path)

    save_path = 'test.wav'
    wavfile.write(save_path, FS, waveform.astype(np.int16))
