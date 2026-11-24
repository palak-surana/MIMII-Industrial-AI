"""Band-pass filter + log-Mel spectrogram (numpy/scipy only)."""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Returns (filterbank [n_mels, n_fft//2+1], hz_edges [n_mels+2])."""
    fmax = fmax or sr / 2
    hz = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        l, c, r = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.maximum(0, np.minimum((bins - l) / (c - l), (r - bins) / (r - c)))
    fb *= (2.0 / (hz[2:] - hz[:-2]))[:, None]
    return fb.astype(np.float32), hz


def bandpass(x, sr, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


def log_mel(x, n_fft, hop, fb):
    """x: 1-D signal -> log-Mel spectrogram in dB, shape (n_mels, n_frames)."""
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(xp[idx] * np.hanning(n_fft), axis=1)) ** 2
    mel = spec @ fb.T
    return (10.0 * np.log10(np.maximum(mel, 1e-10))).T.astype(np.float32)
