# File containing function that can compute mel-spectrograms and mel-coefficients
# Based on: http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

import numpy as np
from scipy.fftpack import dct


def melSpectrogram(signal, sample_rate, frame_size=0.025, frame_stride=0.01, nfilt=70, NFFT=1024, normalized=True):
    """
    Computes the mel-spectrogram of an input signal

    :param signal: 1d-array, signal of the sound extract
    :param sample_rate: int, sample rate of the sound extract
    :param frame_size: float [0.025], size of the Fourier-transform window
    :param frame_stride: float [0.01], size of the stride during Fourier-transform
    :param nfilt: int [70], number of frequency bins
    :param NFFT: int [1024], number of points used for the Fourier-transform
    :param normalized: bool [True], indicates whether to apply mean-removal or not

    :return: 2d-array, the spectrogram
    """

    # Pre - Emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # Framing
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(
        float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal,
                           z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Window
    frames *= np.hamming(frame_length)

    # Fourier - Transform and PowerSpectrum
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    # Filter Banks
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    # Mean-removal
    if normalized:
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        # filter_banks /= (np.std(filter_banks) + 1e-12)

    return filter_banks


def mfcc(signal, sample_rate, frame_size=0.025, frame_stride=0.01, nfilt=70, NFFT=1024, normalized=True):
    """
    Computes the Mel-frequency Cepstral Coefficients (MFCCs)

    :param signal: 1d-array, signal of the sound extract
    :param sample_rate: int, sample rate of the sound extract
    :param frame_size: float [0.025], size of the Fourier-transform window
    :param frame_stride: float [0.01], size of the stride during Fourier-transform
    :param nfilt: int [70], number of frequency bins
    :param NFFT: int [1024], number of points used for the Fourier-transform
    :param normalized: bool [True], indicates whether to apply mean-removal or not

    :return: 2d-array, the mel-frequency coefficients
    """

    cep_lifter = 22
    num_ceps = 12

    # Filter banks
    filter_banks = melSpectrogram(signal, sample_rate, frame_size, frame_stride, nfilt, NFFT, normalized)

    # Discrete Cosine Transform
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13

    # Sinusoidal liftering
    nframes, ncoeff = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift

    # Mean-removal
    if normalized:
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return mfcc
