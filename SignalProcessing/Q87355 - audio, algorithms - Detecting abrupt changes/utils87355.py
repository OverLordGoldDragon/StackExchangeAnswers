# -*- coding: utf-8 -*-
import os
import numpy as np
from numpy.fft import (fft2 as nfft2, ifft2 as nifft2, fft as nfft,
                       ifft as nifft, ifftshift as nifftshift)
import matplotlib.pyplot as plt

from ssqueezepy import Wavelet, ssq_cwt, ssq_stft, cwt, plot
from pathlib import Path
from scipy.io import wavfile
# from scipy.signal.windows import dpss

__all__ = [
    'wav_cfgs',
    'sparse_mean',
    'cc_2d1d',
    'pad_input_make_t',
    'make_unpad_shorthand',
    'make_impulse_response',
    'handle_wavelet',
]

# Configurables ##############################################################
# data -----------------------------------------------------------------------
SUFFIX = '.wav'
data_dir = Path(__file__).parent.resolve()
data_labels = {
    "example":       (1.55, 3.5),
    "T2_T_00043244": (1.21, 1.74),
    "T2_T_00043246": (0.94, 1.57),  # note: was idx 3
    "T2_D_00004326": (1.42, 1.85),  # note: was idx 6

    "T2_T_00043245": (0.76, 3.38),  # note: was idx 2
    "T2_D_00004323": (0.79, 1.68),  # note: was idx 4
    "T2_D_00004324": (2.87, 3.28),
    "T2_D_00004328": (0.75, 3.15),
    "T2_D_00004330": (0.63, 3.24),
    "T2_D_00004331": (0.46, 2.02),
    "T2_D_00004333": (4.97, 5.86),
    "T2_D_00004335": (0.94, 2.09),
    "T2_D_00004338": (2.37, 2.99),
    "T2_D_00004339": (2.03, 2.73)
}
data_names = list(data_labels)

# algorithm ------------------------------------------------------------------
wav_cfgs = {'time': 20, 'balanced': 60, 'freq': 140}

# misc -----------------------------------------------------------------------
vline_cfg = {'color': 'tab:red', 'linewidth': 2}

# Backend ####################################################################
IS_GPU = bool(os.environ.get('SSQ_GPU', None) == '1')
IS_FULL_GPU = bool(os.environ.get('FULL_GPU', None) == '1')
MOVE_TO_CPU = bool(IS_GPU and not IS_FULL_GPU)
if IS_GPU:
    import torch
    from torch.fft import (fft2 as tfft2, ifft2 as tifft2, fft as tfft,
                           ifft as tifft, ifftshift as tifftshift)

if IS_FULL_GPU:
    raise NotImplementedError

def _handle_device(x):
    if IS_GPU and isinstance(x, np.ndarray):
        x = torch.as_tensor(x, device='cuda')
    return x

def _handle_fn(fn, *a, **k):
    x = _handle_device(a[0])
    return fn(x, *a[1:], **k)

def fft2(*a, **k):
    fn = tfft2 if IS_GPU else nfft2
    return _handle_fn(fn, *a, **k)

def ifft2(*a, **k):
    fn = tifft2 if IS_GPU else nifft2
    return _handle_fn(fn, *a, **k)

def fft(*a, **k):
    return nfft(*a, **k)

def ifft(*a, **k):
    return nifft(*a, **k)

def ifftshift(*a, **k):
    return nifftshift(*a, **k)

# Data helpers ###############################################################
def load_data(example_index):
    """Returns `x, fs, labels`."""
    data_name = data_names[example_index]
    path = str(Path(data_dir, data_name)) + SUFFIX

    # handle `x`, `fs`
    fs, x = wavfile.read(path)
    x = (x / abs(x).max()).astype('float32')

    # handle `labels`
    labels = list(data_labels[data_name])
    # now filter it to exclude endpoint (lack-of-change substitute)
    duration = len(x) / fs
    if abs(labels[1] - duration) < 0.1:
        labels.pop()
    labels = tuple(labels)

    return x, fs, labels


# Feature helpers ############################################################
def sparse_mean(x, div=100, iters=4):
    """Mean of non-negligible points"""
    m = x.mean()
    for _ in range(iters - 1):
        m = x[x > m / div].mean()
    return m

def cc_2d1d(x, hf):
    M, N = x.shape
    prod = fft2(x) * _handle_device(hf)

    sub = M
    xfs = prod.reshape(sub, -1, N).mean(axis=0)

    out = ifft2(xfs).real
    if MOVE_TO_CPU:
        out = out.cpu()
    return out


def pad_input_make_t(x, fs, N_ref=None):
    if N_ref is None:
        N_ref = len(x)
    t = np.linspace(0, len(x)/fs, len(x))
    # right-pad by at least the length of `x`, but to power of 2,
    # then not at all internally
    padded_len_min = 2*N_ref
    padded_len = int(2**np.ceil(np.log2(padded_len_min)))
    pad_right = (padded_len - len(x))//2
    pad_left = padded_len - len(x) - pad_right
    xp = np.pad(x, [pad_left, pad_right])
    tp = np.pad(t, [pad_left, pad_right])

    return xp, tp, pad_left, pad_right


def make_unpad_shorthand(pad_left, pad_right):
    def u(x):
        return x[..., pad_left:-pad_right]
    return u


def make_impulse_response(ssq_cfg, fmax_idx, escale, escaling):
    M = ssq_cfg['wavelet']._Psih.shape[-1]
    ir2d = np.zeros(M)
    ir2d[M//2] = 1
    ir2d = abs(cwt(ir2d, **ssq_cfg, astensor=False)[0])[:fmax_idx]
    ir2d /= ir2d.max(axis=-1)[:, None]

    if escaling[0]:
        ir2d *= escale**2
    ir2df = np.conj(nfft2(ifftshift(ir2d, axes=1)))
    return ir2d, ir2df


def handle_wavelet(wavelet, fs, M, ssq_precfg, fmax_idx_frac,
                   silence_interval_samples, escaling):
    # arg checks
    is_ssqueezepy_wavelet = lambda wavelet: bool(
        wavelet.__class__.__name__ == 'Wavelet')
    if isinstance(wavelet, tuple) and is_ssqueezepy_wavelet(wavelet[0]):
        return wavelet
    elif is_ssqueezepy_wavelet(wavelet):
        pass
    elif isinstance(wavelet, str):
        assert wavelet in ('freq', 'time', 'balanced')
    else:
        raise ValueError("invalid `wavelet`")

    if isinstance(wavelet, str):
        wav_cfg = ('gmw', {'beta': wav_cfgs[wavelet]})
        wavelet = Wavelet(wav_cfg)

    # generate sampled wavelets array, and scales ----------------------------
    _, scales = cwt(np.arange(M), wavelet, **ssq_precfg)

    # handle `fmax_idx_frac`
    fmax_idx = int(round(fmax_idx_frac * len(scales)))
    # make `ssq_cfg`
    ssq_precfg = ssq_precfg.copy()  # don't affect external copy
    del ssq_precfg['scales']
    ssq_cfg = dict(wavelet=wavelet, scales=scales, **ssq_precfg)

    # handle `escaling` ------------------------------------------------------
    if any(escaling):
        escale = (np.logspace(np.log10(1), np.log10(np.sqrt(.1)), fmax_idx
                              )**2)[:, None]
    else:
        escale = None

    # make impulse response --------------------------------------------------
    _, ir2df = make_impulse_response(ssq_cfg, fmax_idx, escale, escaling)

    # generate energy-aggregating window -------------------------------------
    # approximate the effective temporal width of the wavelet with median
    # temporal width
    psihs = wavelet._Psih
    # fetch median wavelet, take it to time
    pf = psihs[int(.42 * len(psihs))]  # .42 = long story
    if MOVE_TO_CPU:
        pt = abs(tifft(pf).cpu())
    else:
        pt = abs(nifft(pf))
    # compute its eff two-sided temporal width
    pwidth = np.where(pt < pt.max()/4)[0][0] * 2
    # make the window
    wsummer = np.zeros(M)
    wsummer[:pwidth//2] = 1
    wsummer[-(pwidth//2 - 1):] = 1
    wsummerf = fft(wsummer)

    # generate silence-detecting window --------------------------------------
    if silence_interval_samples > 0:
        wsilence = np.zeros(M)
        wsilence[:silence_interval_samples//2 + 1] = 1
        wsilence[-(silence_interval_samples//2 - 1):] = 1
        wsilencef = fft(wsilence)
    else:
        wsilencef = None

    # return -----------------------------------------------------------------
    reusables = (wavelet, ssq_cfg, ir2df, pwidth, wsummerf, wsilencef, escale,
                 fmax_idx)
    return reusables


def derivative_along_freq(Tx_slc):
    Tx_slc_d = Tx_slc.copy()
    Tx_slc_d = abs(np.diff(Tx_slc, axis=0))

    # pad to restore sample such that
    # `diff(x)[n] == x[n] - x[n - 1]` for all `n` except `diff(x)[0] == 0`.
    Tx_slc_d = np.pad(Tx_slc_d, [[1, 0], [0, 0]])
    return Tx_slc_d

# Scoring helpers ############################################################
def _get_predictions_around_peak(g_slc, slc, pwidth):
    peak = np.argmax(g_slc)
    g_slc = g_slc.copy()
    g_slc[peak - pwidth:peak + pwidth] = 0
    second_peak = np.argmax(g_slc)

    # finalize and shift back to original coordinates
    one_peak_pred = peak + slc.start
    two_peaks_pred = (peak + second_peak) // 2 + slc.start
    return one_peak_pred, two_peaks_pred


def _pred_not_near_silence(xsilence, pred, silence_proximity_samples):
    start = max(pred - silence_proximity_samples, 0)
    end = pred + silence_proximity_samples
    return bool(xsilence[start:end].max() == 0)


def get_predictions(g, tp, xsilence, pwidth, min_audio_interval_samples,
                    silence_proximity_samples, final_pred_n_peaks, n_labels):
    g = g.copy()
    preds_all = []  # final predictions
    preds_all_nofail = []  # if both haven't failed

    pred_favor_idx = (0 if final_pred_n_peaks == 1 else
                      1)

    while len(preds_all) < n_labels:
        # get peak and carve it out
        peak = np.argmax(g)
        slc = slice(peak - min_audio_interval_samples//2,
                    peak + min_audio_interval_samples//2)
        g_slc = g[slc].copy()
        g[slc] = 0
        # predict around interval
        preds = _get_predictions_around_peak(g_slc, slc, pwidth)

        # check for proximity with silence
        if silence_proximity_samples > 0:
            one_peak_valid, two_peaks_valid = [
                _pred_not_near_silence(xsilence, pred, silence_proximity_samples)
                for pred in preds]
        else:
            one_peak_valid, two_peaks_valid = True, True

        # append preds if applicable
        if not (one_peak_valid or two_peaks_valid):
            # if nothing is valid, continue with `g` carved out around
            # current `peak`
            continue
        else:
            # convert to seconds, append
            preds = tuple(tp[pred] for pred in preds)
            if one_peak_valid and two_peaks_valid:
                preds_all_nofail.append(preds)
                preds_all.append(preds[pred_favor_idx])
            elif one_peak_valid:
                preds_all_nofail.append((preds[0], None))
                preds_all.append(preds[0])
            elif two_peaks_valid:
                preds_all_nofail.append((None, preds[1]))
                preds_all.append(preds[1])

        # avoid infinite loop
        if np.argmax(g) == 0:
            raise Exception("carved out entire feature vector without making "
                            "enough predictions! Try lowering `silence` "
                            "variables.")

    return preds_all, preds_all_nofail



# The complete algorithm #####################################################


def find_audio_change_timestamps(
        x, reusables=None, fs=1, wavelet='freq', stft_prefilter_scaling=1,
        silence_interval=.2, silence_proximity=.2, silence_th_scaling=4,
        min_audio_interval=.5,
        impulse_pass=True, escaling=(1, 1, 1), pwidth_scaling=1, fmax_idx_frac=1,
        carve_th_scaling=1/3, n_carvings=2, final_pred_n_peaks=1, n_labels=2,
        viz_labels=None, N_ref=None):
    """Returns `n_labels` predictions, sorted as "earliest = likeliest".

    Parameters
    ----------
    x : np.ndarray
        1D float array

    wavelet : str / tuple
        str:
            "freq" for high frequency resolution
            "time" for high time resolution
            "balanced" for balanced resolution

        tuple:
            output of `handle_wavelet` or one of outputs of this function

    stft_prefilter_scaling : float
        0 to disable

    fs : int
        Sampling rate.
        Used only to handle `silence_interval` and `min_audio_interval`.

    silence_interval : float
        Will not return predictions over intervals of silence.
        In seconds. `0` to disable.

    silence_proximity : float
        Will check if, `silence_proximity` seconds within a prediction,
        there is an interval of silence that lasts `silence_interval` seconds.
        No effect if `silence_interval=0`.

    silence_th_scaling : float
        Multiplies thresholding used to detect "silence"; higher means more
        points are identified as silence.

    min_audio_interval : float
        Minimum possible interval between labels.
        In seconds. `0` to disable.

    impulse_pass : bool (default True)
        Whether to do the impulse pass.

    escaling : bool (default True)
        Quadratic energy scaling favoring higher frequencies, applied upon
        doubly carved `Tx` before making the final 1D feature vector.

    pwidth_scaling : float (default 1)
        Wavelet median width scaling, used for aggregating energies for the final
        1D feature vector.

    fmax_idx_frac : float (default 1)
        Whether to use a subset of all SSQ frequencies, starting from high freqs
        (so only lows can ever be excluded).

    carve_th_div : float
        Carving sparse mean divisor.

    n_carvings : int (default 2)
        Number of times to carve out the derivative.

    final_pred_n_peaks : 1
        Which peaks prediction to favor if both are available, one-peak or
        two-peak.

    n_labels : int (default 2)
        Number of predictions to return. Works like "top k".

    Returns
    -------
    preds_all, preds_all_nofail, reusables
    """
    # apply `fs`: convert seconds to samples
    (silence_interval_samples, silence_proximity_samples,
     min_audio_interval_samples) = [
         int(h*fs) for h in
         (silence_interval, silence_proximity, min_audio_interval)]

    # handle `x`
    xp, tp, pad_left, pad_right = pad_input_make_t(x, fs, N_ref)
    u = make_unpad_shorthand(pad_left, pad_right)
    M = len(xp)

    # handle `wavelet`
    ssq_precfg = dict(scales='log', padtype=None)
    if reusables is None:
        reusables = handle_wavelet(wavelet, fs, M, ssq_precfg, fmax_idx_frac,
                                   silence_interval_samples, escaling)
    (wavelet, ssq_cfg, ir2df, pwidth, wsummerf, wsilencef, escale, fmax_idx
     ) = reusables

    # make silence vector, if specified --------------
    if silence_interval_samples > 0:
        ex = np.abs(xp)**2
        exw = ifft(wsilencef * fft(ex)).real
        # set regions within influence of padding to maximum
        exw[:pad_left + silence_interval_samples//2] = exw.max()
        exw[-(pad_right + silence_interval_samples//2):] = exw.max()
        # make thresholding then compute final silence vector
        silence_th = sparse_mean(u(ex)) * silence_th_scaling
        xsilence = (exw < silence_th).astype(bool)

    # pre-filtering with stft ------------------------
    if stft_prefilter_scaling > 0:
        win = 'hamm'
        #win = dpss(512, 512//128)
        n_fft = int(512 * (fs / 16000))  # match duration used during experiments
        Tsx = ssq_stft(x, win, n_fft=n_fft, flipud=True, astensor=False)[0]
        aTsx_slc = abs(Tsx)
        if escaling[1]:
            escale_lin = np.linspace(1, 0, len(Tsx))[:, None]
            aTsx_slc *= escale_lin
        aTsx_slc = aTsx_slc**2
        aTsx_slc = aTsx_slc[:int(.75*len(Tsx))]

        stft_carve_th = sparse_mean(aTsx_slc) * stft_prefilter_scaling
        Tsx[:len(aTsx_slc)][aTsx_slc > stft_carve_th] = 0
        xfilt = Tsx.sum(axis=0).real

        xp, tp, pad_left, pad_right = pad_input_make_t(xfilt, fs, N_ref)

    # transform --------------------------------------
    Txp, *_ = ssq_cwt(xp, **ssq_cfg, astensor=False)
    Tx_slc = abs(Txp[:fmax_idx])

    # do carving -------------------------------------
    # applies `n_carvings`, `carve_th_div`
    if escaling[1]:
        Tx_slc *= escale
    for _ in range(n_carvings):
        Tx_slc_d = derivative_along_freq(Tx_slc)
        carve_th = sparse_mean(Tx_slc_d) * carve_th_scaling
        Tx_slc[Tx_slc_d > carve_th] = 0

    # carve post-processing
    # applies `escaling` and `silence_interval`
    if escaling[2]:
        Tx_slc *= escale
    if silence_interval_samples > 0:
        Tx_slc[:, xsilence] = 0

    # compute 1D feature vector ----------------------
    # applies `impulse_pass`
    if impulse_pass:
        g = cc_2d1d(Tx_slc, ir2df)[0]
        g = g**2
    else:
        g = g**2
        g = g.sum(axis=0)
    # the "subtle impulse intensity" vector
    g = ifft(wsummerf * fft(g)).real

    if viz_labels is not None:
        example_index, labels = viz_labels
        fig, ax = plt.subplots(constrained_layout=True)
        plot(u(tp), u(g), vlines=(list(labels), vline_cfg),
             title="{}: {}".format(example_index, labels), fig=fig, ax=ax,
             w=.6, h=.7)
        fig.savefig(Path(data_dir, "im{}.png".format(example_index)))
        plt.show()

    # make predictions -------------------------------
    preds_all, preds_all_nofail = get_predictions(
        g, tp, xsilence, pwidth, min_audio_interval_samples,
        silence_proximity_samples, final_pred_n_peaks, n_labels)

    # return -----------------------------------------
    return preds_all, preds_all_nofail, reusables
