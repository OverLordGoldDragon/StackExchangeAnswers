# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/76329/50076 ################################
import numpy as np
from numpy.fft import ifft, ifftshift
from ssqueezepy.visuals import plot, scat, imshow
from ssqueezepy.utils import make_scales, cwt_scalebounds
from ssqueezepy import ssq_cwt, cwt, Wavelet
from kymatio.scattering1d.filter_bank import morlet_1d

#%%# Helper methods ##########################################################
def viz0(xf, xref=None, center=True):
    x = ifft(xf)
    if center:
        x = ifftshift(x)
    scat(xf[:32], show=1)
    if xref is not None:
        xref *= np.abs(x).max() / np.abs(xref).max()
        plot(xref, color=grey, linestyle='--')
    plot(x.real, show=1)

def morlet(N, xi, sigma=.1/2, viz=1, trim=True, nonhalved=False):
    xf = morlet_1d(N, xi=xi, sigma=sigma)

    trim_idx = int(xi * N) + 1
    if trim:
        xf[trim_idx:] = 0
    if not nonhalved:
        xf[trim_idx - 1] /= 2
    x = ifftshift(ifft(xf))

    if viz:
        scat(xf, show=1)
        plot(x, complex=1)
        plot(x, abs=1, color='k', linestyle='--', show=1)
    return x, xf

def ref_sine(xi, x=None, N=None, endpoint=False, zoom=None):
    N = N if N is not None else len(x)
    t = np.linspace(0, 1, N, endpoint=endpoint)
    xref = np.cos(2*np.pi * (xi * N) * t)
    if x is not None:
        xref *= x.real.max() / xref.max()

    if zoom is not None:
        ctr = N // 2
        zoom = (zoom, zoom) if not isinstance(zoom, tuple) else zoom
        a, b = ctr - zoom[0], ctr + zoom[1] + 1
        _t = np.arange(a, b)

        plot(_t, xref[a:b], color=grey, linestyle='--')
        if x is not None:
            plot(_t, x.real[a:b], show=1, title="zoomed, real part")
    return xref

grey = (.1, .1, .1, .4)

#%%# Simple sinusoid #########################################################
N = 128
t = np.linspace(0, 1, N, 0)
xf = np.zeros(N)

xf[16] = 1
xref = ifft(xf).real
viz0(xf, center=0)

#%%# Add lateral peaks
xf = np.zeros(N)
xf[15] = .5
xf[17] = .5
viz0(xf, center=0)
#%%# Both at once
xf[16] = 1
viz0(xf, xref)
#%%# Positive + negative example
xf = np.zeros(N)
xf[15] = .5
xf[17] = -.5
viz0(xf, center=0)

#%%# GIF #####################################################################
xf_full = morlet_1d(N, xi=16/128, sigma=.1/8)
xf = np.zeros(N)
xf[16] = xf_full[16]
viz0(xf)

for idx in (17, 18, 19, 20, 21):
    xf[idx] = xf_full[idx]
    xf[16 + (16 - idx)] = xf_full[16 + (16 - idx)]
    viz0(xf, xref)

#%%# CWT near Nyquist ########################################################
N = 129
t = np.linspace(0, 1, N, 1)
x = np.cos(2*np.pi * 64 * t)

wavelet = Wavelet('morlet')
min_scale, max_scale = cwt_scalebounds(wavelet, N=N, preset='minimal')
# adjust such that highest freq wavelet's peak is at Nyquist
scales0 = make_scales(N, min_scale, max_scale, wavelet=wavelet) * 1.12
scales = scales0
Wx, _ = cwt(x, wavelet, scales=scales)

imshow(Wx, abs=1, title="abs(CWT)", xlabel="time", ylabel="scale",
       yticks=scales)
plot(Wx[:, 0], abs=1, title="abs(CWT) at a time slice (same for all slices)",
     xlabel="scale", show=1, xticks=scales)
imshow(Wx.real)

#%%# Morlet near nyquist #####################################################
N = 128
xf = morlet(N, xi=.5, viz=1)

#%%# Trimmed peak but higher sampling rate
N = 512
xi = .5 / 4
x, xf = morlet(N, xi=xi, sigma=.1/4, viz=1, nonhalved=True)
x, xf = morlet(N, xi=xi, sigma=.1/4, viz=1)

#%%# Zoomed sinusoid at center frequency
xref = ref_sine(xi, x=x, zoom=20)
xref = ref_sine(xi, x=x, zoom=(80, -40))

#%%# SSQ_CWT of wavelet ######################################################
# extreme time-localization
wavelet1 = Wavelet(('gmw', {'beta': 1, 'gamma': 1}))

Tx0, Wx0, *_ = ssq_cwt(x.real, wavelet1, scales='log')
mx = np.abs(Tx0).max() * .6
# zoomed
imshow(Tx0, abs=1, title="abs(SSQ_CWT) of wavelet.real", norm=(0, mx))
imshow(Tx0[20:160], abs=1, norm=(0, mx))

#%%# CWT
# to take perfect CWT of wavelet, tweak slightly to account for padding
xref = np.cos(2*np.pi * (N//8) * np.linspace(0, 1, N + 1, 1))
Tx1, Wx1, *_ = ssq_cwt(xref, wavelet, scales='log')
imshow(Tx1, abs=1, title="abs(SSQ_CWT) of sinusoid at peak center freq")

#%%# Peak center frequency ###################################################
N = 512
xi = .5 / 4
x, xf = morlet(N, xi=xi, sigma=.1/4, viz=0, nonhalved=True)
scat(xf)
f = int(N * xi)
scat(np.array([f]), xf[f], color='red', s=30, show=1)

#%%
xref = ref_sine(xi, x=x)
plot(xref, color=grey)
plot(x.real, show=1, title="peak center frequency")

#%%
t = np.linspace(0, 1, N, 1)
xref = np.cos(2*np.pi * 64 * np.linspace(0, 1, 513, 1))

wavelet = Wavelet('morlet')
min_scale, max_scale = cwt_scalebounds(wavelet, N=N, preset='minimal')
scales = make_scales(N, min_scale, max_scale, wavelet=wavelet) * 1.12
Wx, scales = cwt(xref, wavelet, scales=scales)

#%%# Energy center frequency #################################################
axf = np.abs(xf)
axfs = axf**2
fmean_l1 = np.sum(np.arange(len(x)) * axf / axf.sum())
fmean = np.sum(np.arange(len(x)) * axfs / axfs.sum())
xref = np.cos(2*np.pi * fmean * np.linspace(0, 1, N, 0))
xref *= x.real.max() / xref.max()

plot(xref, color=grey)
plot(x.real, show=1, title="energy center frequency")

#%% For spectrum use one near Nyquist
xref = ref_sine(xi=.5, N=129, endpoint=1)
Wx, _ = cwt(xref, 'morlet', scales=scales0)
imshow(Wx, abs=1)

slc = np.abs(Wx[:, 0])
fmean_nyq = np.sum(np.arange(len(slc)) * (slc**2) / (slc**2).sum())
scat(slc, vlines=(fmean_nyq, {'color': 'tab:red'}), show=1)
imshow(Wx, abs=1)

#%%# Central instantaneous center frequency ##################################
# determine central instantaneous frequency from CWT since SSQ reassigns
# nonlinearly
N = len(Tx0[0])
finst_idx = np.argmax(np.abs(Wx0[:, 256]))
psihs = wavelet1.Psih()
finst = np.argmax(np.abs(psihs[finst_idx])) // 2  # x2 pad doubles index

#%%
xref = np.cos(2*np.pi * finst * np.linspace(0, 1, N, 0))
xref *= x.real.max() / xref.max()
ctr = N // 2
d = 9
a, b = ctr - d, ctr + d + 1
xref = xref[a:b] * N / (a - b)
_t = np.arange(a, b)

plot(_t, xref, color=grey, auto_xlims=0)
plot(x.real, show=1, title="central instantaneous center frequency")
