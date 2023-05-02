# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/87705/50076
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import scipy.signal

try:
    from ssqueezepy.visuals import plotscat as _plotscat, plot
except:
    print("Plotting requires ssqueezepy; won't plot")
    _plotscat = lambda *a, **k: 1
    plot = lambda *a, **k: 1

def plotscat(*a, **k):
    x = a[0]
    if k.get('abs', 0):
        x = abs(x)
    mn, mx = x.min(), x.max()
    amx = 1.03*max(abs(mn), abs(mx))
    ylims = (-amx, amx) if mn < -.03 else (0, amx)
    _plotscat(x, *a[1:], **k, ylims=ylims)


def energy(x):
    return np.sum(abs(x)**2)

def get_xf_sub_worst(xf, M):
    return (abs(xf.real) + 1j*abs(xf.imag)).reshape(M, -1).mean(axis=0)

def measure(xf, M):
    N = len(xf)
    hbw = N//M//2
    xf_ref = np.zeros(N)
    xf_ref[:hbw] = 1
    xf_ref[-hbw:] = 1

    xf /= abs(xf).max()  # in case this step wasn't done already

    xf_sub = get_xf_sub_worst(xf, M)
    xf_ref_sub = xf_ref.reshape(M, -1).mean(axis=0)

    E_x_before   = energy(xf)
    E_ref_before = energy(xf_ref)
    E_x_after    = energy(xf_sub)
    E_ref_after  = energy(xf_ref_sub)
    ratio_before = E_x_before / E_ref_before
    ratio_after  = E_x_after  / E_ref_after
    r = ratio_after / ratio_before

    alias = round(100*(r - 1) / (M - 1), 3)
    return alias

#%% Make signals
N = 256
M = 8

# populate unaliased signal
hbw = N//M//2
xf_ref = np.zeros(N)
xf_ref[:hbw] = 1
xf_ref[-hbw:] = 1

# populate referenced signal
xf_full = np.ones(N)

# make in-between signal
xf0 = xf_ref.copy()
xf0[hbw*2] = 1
xf0[-(hbw*2)] = 1

#%% Plot
def viz(xf, name, aval=0, do_measure=0, worst=0, cval=0):
    fig, axes = plt.subplots(1, 2, layout='constrained', figsize=(12, 5))
    pkw = dict(fig=fig, abs=aval, complex=cval)
    if worst:
        sub = get_xf_sub_worst(xf, M)
    else:
        sub = xf.reshape(M, -1).mean(axis=0)
    l, r = ("|", "|") if aval else ("", "")
    info = ("abs_max={:.3g}".format(max(abs(sub))) if not do_measure else
            "alias={:.3g}%".format(measure(xf, M)))
    title1 = "{}fft({}){}".format(l, name, r)
    if worst:
        title2 = "{}fft({}_sub{}_worst){} -- {}".format(l, name, M, r, info)
    else:
        title2 = "{}fft({}[::{}]){} -- {}".format(l, name, M, r, info)

    plotscat(xf,  **pkw, ax=axes[0], title=title1)
    plotscat(sub, **pkw, ax=axes[1], title=title2)
    plt.show()

viz(xf_ref, "x_ref")
viz(xf_full, "x_full")
viz(xf0, "x0")

#%% Applying the metric
def scipy_decimate_filter(N, M):
    """Minimally reproduce the filter used by
    `scipy.signal.decimate(ftype='fir')`. This is rigorously tested elsewhere.
    """
    q = M
    half_len = 10*q
    n = int(2*half_len)
    cutoff = 1. / q
    numtaps = n + 1

    win = scipy.signal.get_window("hamming", numtaps, fftbins=False)

    # sample, window, & norm sinc
    alpha = 0.5 * (numtaps - 1)
    m = np.arange(0, numtaps) - alpha
    h = win * cutoff * np.sinc(cutoff * m)
    h /= h.sum()  # L1 norm

    # pad and center
    h = np.pad(h, [0, N - len(h)])
    h = np.roll(h, -np.argmax(h))
    return h

# moving average
h = np.zeros(N)
h[:M//2 + 1] = 1/M
h[-(M//2 - 1):] = 1/M
hf = fft(h)

# scipy
h_scipy = scipy_decimate_filter(N, M)
hf_scipy = fft(h_scipy)

#%% Show results
viz(hf, "h", aval=1, do_measure=1, worst=1)
viz(hf_scipy, "h_scipy", aval=1, do_measure=1, worst=1)

#%% Effect
def dft_upsample(xf, M):
    L = len(xf)
    n_to_add = L * M - L
    zeros = np.zeros(n_to_add - 1)
    nyq = xf[L//2]
    return np.hstack([xf[:L//2],
                      nyq/2, zeros, np.conj(nyq)/2,
                      xf[-(L//2 - 1):]]) * M
np.random.seed(42)
x = np.random.randn(N)
x0_nosub = ifft(fft(x) * hf)
x1_nosub = ifft(fft(x) * hf_scipy)
_x0f = fft(x0_nosub).reshape(M, -1).mean(axis=0)
_x1f = fft(x1_nosub).reshape(M, -1).mean(axis=0)

x0f = dft_upsample(_x0f, M)
x1f = dft_upsample(_x1f, M)
x0, x1 = ifft(x0f), ifft(x1f)


fig, axes = plt.subplots(1, 2, layout='constrained', figsize=(12, 5),
                         sharey=True)
pkw = dict(fig=fig)
plot(x0_nosub.real, ax=axes[0])
plot(x0.real, ax=axes[0], title="hf: recovered")
plot(x1_nosub.real, ax=axes[1])
plot(x1.real, ax=axes[1], title="hf_scipy: recovered")
