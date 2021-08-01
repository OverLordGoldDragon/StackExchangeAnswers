# https://dsp.stackexchange.com/q/76560/50076 ################################
# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import ifft, ifftshift
from kymatio.scattering1d.filter_bank import morlet_1d
from kymatio.visuals import plot as kplot, plotscat

def plot(*args, **kw):
    if 'title' in kw:
        kw['title'] = (kw['title'], {'fontsize': 18})
    mx = re_im_max(args[0]) * 1.05
    kw['ylims'] = (-mx, mx)
    kplot(*args, **kw)

def re_im_max(x):
    return max(np.abs(x.real).max(), np.abs(x.imag).max())

def re_im_parts(x, y):
    A =  x.real * y.real
    B = -x.imag * y.imag
    C =  x.real * y.imag
    D =  x.imag * y.real
    return A, B, C, D

def l2(x):
    return np.sqrt(np.sum(np.abs(x)**2))

#%%# Morlet visuals ##########################################################
N = 256
pf = morlet_1d(N, xi=.025, sigma=.1/20)
pt = ifftshift(ifft(pf))
#%%
kw = dict(complex=1, show=1, ticks=(1, 0))
plot(pt, **kw, title="psi: analytic Morlet")

#%%
A, B, C, D = re_im_parts(pt, pt)
plot(A + 1j*B, **kw, title="(psi * psi).real")
plot(C + 1j*D, **kw, title="(psi * psi).imag")
plot(pt * pt,  **kw,
     title="(psi * psi).real + (psi * psi).imag = psi * psi")

#%%
pta = np.conj(pt)
plot(pta, **kw, title="psia: anti-analytic Morlet")

#%%
plot(pt.imag + 1j*pta.imag, **kw, title="psia = psi.real - psi.imag")

#%%
A, B, C, D = re_im_parts(pt, pta)
plot(A + 1j*B, **kw, title="(psi * psia).real")
plot(C + 1j*D, **kw, title="(psi * psia).imag")
plot(pt * pta, **kw,
     title="(psi * psia).real + (psi * psia).imag = psi * psia")

#%%# Random sequence visuals #################################################
def viz_seq(x, show=1, complex=1, **kw):
    mx = re_im_max(x) * 1.1
    plotscat(x, complex=complex, ylims=(-mx, mx), show=show, **kw,
             hlines=(0, {'color': 'tab:red', 'linewidth': 1}),
             ticks=(1, 0), auto_xlims=0)

def zero_sum_seq(N):
    x = np.random.randn(N) + 1j*np.random.randn(N)
    x[N//2:] = 0

    slc = x[:N//2][::-1]
    x[N//2:] = slc.real - 1j * slc.imag
    x -= x.mean()

    x.real *= (l2(x.imag) / l2(x.real))
    return x

#%%
np.random.seed(0)
N = 12
x = zero_sum_seq(N)
y = zero_sum_seq(N)

viz_seq(x,     title="x | sum = {:.2f}".format(x.sum()))
viz_seq(y,     title="y | sum = {:.2f}".format(y.sum()))
viz_seq(x * x, title="x * x | sum = {:.2f}".format((x*x).sum()))
viz_seq(y * y, title="y * y | sum = {:.2f}".format((y*y).sum()))
viz_seq(x * y, title="x * y | sum = {:.2f}".format((x * y).sum()))
