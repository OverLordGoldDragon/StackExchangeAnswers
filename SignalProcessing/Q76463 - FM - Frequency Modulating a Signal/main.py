# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/76463/50076 ################################
import numpy as np
from numpy.fft import fft, ifft
from scipy.io import wavfile
from ssqueezepy import ssq_cwt, Wavelet
from ssqueezepy.visuals import imshow, plot

#%%# Helper methods ##########################################################
def frequency_modulate(slc, fc=None, b=.3):
    N = len(slc)
    if fc is None:
        fc = N / 18  # arbitrary
    # track actual `b` for demodulation purposes
    b_effective = b

    t_min, t_max = start / fs, end / fs
    t = np.linspace(t_min, t_max, N, endpoint=False)
    assert np.allclose(fs, 1 / np.diff(t))

    x0 = slc[:N]
    # ensure it's [-.5, .5] so diff(phi) is b*[-pi, pi]
    x0max = np.abs(x0).max()
    x0 /= (2*x0max)
    b_effective /= (2*x0max)

    # generate phase
    phi0 = 2*np.pi * fc * t
    phi1 = 2*np.pi * b * np.cumsum(x0)
    phi = phi0 + phi1
    diffmax  = np.abs(np.diff(phi)).max()
    # `b` correction
    if diffmax > np.pi or np.allclose(phi, np.pi):
        diffmax0 = np.abs(np.diff(phi0)).max()
        diffmax1 = np.abs(np.diff(phi1)).max()
        # epsilon term for stable inversion / pi-unambiguity
        eps = 1e-7
        factor = ((np.pi - diffmax0 - eps) / diffmax1)
        phi1 *= factor
        b_effective *= factor
        phi = phi0 + phi1
    assert np.abs(np.diff(phi)).max() <= np.pi

    # modulate
    x = np.cos(phi)
    return x, t, phi0, phi1, b_effective

def analytic(x):
    N = len(x)
    xf = fft(x)

    xaf = np.zeros(N, dtype='complex128')
    xaf[:N//2 + 1] = 2 * xf[:N//2 + 1]
    xaf[0] /= 2
    xaf[N//2] /= 2

    xa = ifft(xaf)
    assert np.allclose(xa.real, x)
    return xa

#%%# Load data, select slice #################################################
fs, data = wavfile.read(r"C:\Desktop\recording.wav")
data = data.astype('float64')
data /= (2*np.abs(data).max())

start, end = 0, fs
slc = data[start:end]

#%%# Modulate & validate #####################################################
x, t, phi0, phi10, b_effective = frequency_modulate(slc)

# extreme time localization
wavelet = Wavelet(('gmw', {'gamma': 1, 'beta': 1}))
# synchrosqueezed CWT
Tx, Wx, ssq_freqs, *_ = ssq_cwt(x, wavelet)
# ints for better plotting
ssq_freqs = (ssq_freqs * fs).astype(int)

#%%# Visualize ##############################################################
def viz(t_min=None, t_max=None, f_min=None, f_max=None, show_original=False,
        show_modulated=False):
    freqs = ssq_freqs[::-1]
    a = int(t_min * fs) if t_min is not None else 0
    b = int(t_max * fs) if t_max is not None else None
    d = np.argmin(np.abs(freqs - f_min)) if f_min is not None else None
    c = np.argmin(np.abs(freqs - f_max)) if f_max is not None else 0

    imshow(Tx[c:d, a:b], xticks=t[a:b], yticks=freqs[c:d], **kw)

    if show_original:
        plot(t[a:b], slc[a:b], xlabel="time [sec]", title="original data",
             show=1)
    if show_modulated:
        plot(t[a:b], x[a:b],   xlabel="time [sec]", title="modulated data",
             show=1)

mx = np.abs(Tx).max() * .4
kw = dict(abs=1, norm=(0, mx), title="abs(SSQ_CWT)",
          xlabel="time [sec]", ylabel="frequency [Hz]")
viz()
viz(0, .2,  300, 8000, show_original=1)
viz(0, .02, 300, 8000, show_original=1, show_modulated=1)

#%%
# plot(x, show=1)
# plot(slc, show=1)
phie = np.arccos(x)[:2048]

phi = np.unwrap(np.angle(analytic(x)))[:2048]
phi_exact = (phi0 + phi10)[:2048]
phi[:10] = phi_exact[:10]
ae = np.abs(phi - phi_exact)
print(ae.mean(), ae.max(), ae.min())

phi1 = (phi - phi0[:2048])
x_inv_cs = phi1 / (2*np.pi * b_effective)
x_inv = np.diff(x_inv_cs)
plot(x_inv)
