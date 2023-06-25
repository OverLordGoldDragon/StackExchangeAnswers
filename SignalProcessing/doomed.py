# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/87926/50076
import numpy as np
from numpy.fft import fft

#%%###########################################################################
# The Method
# ----------
def dft_windowed_sine(s, f0, T, tau, n_periods=5):
    # prepare to compute -----------------------------------------------------
    # inputs are continuous-only; determine implied number of samples
    N = T / s
    assert N.is_integer()
    N = int(N)
    # constant scaling
    const = T / (2*s)
    # output indices
    k = np.arange(N)
    # equivalently continuous-time frequencies
    f = k / T
    # common scaling
    c2pi    = -1j*2*np.pi
    # prepare to populate
    out = np.zeros(N, dtype='complex128')
    sum_sweep = range(-(n_periods // 2), n_periods//2 + 1)

    # compute ----------------------------------------------------------------
    # `T` instead of `pi*T` since `np.sinc(x) == sin(pi*x)/(pi*x)`
    for l in sum_sweep:
        f_arg_m = f - l/s - f0
        f_arg_p = f - l/s + f0

        dirac_left  = np.exp(c2pi * ((T - s)/2 * f_arg_m - tau*f0)
                             ) * np.sinc(T * f_arg_m)
        dirac_right = np.exp(c2pi * ((T - s)/2 * f_arg_p + tau*f0)
                             ) * np.sinc(T * f_arg_p)
        out += dirac_left + dirac_right
    out *= const

    return out


def dft_windowed_sine_user(f0, shift, t_total, N_windowed, n_periods=20):
    # what's used in the formula, by definition
    N = N_windowed
    # sampling period, careful not to use `N`
    s = t_total[1] - t_total[0]
    # duration, by definition
    T = N * s

    # here we account for the fact that the derivation was for the window
    # centered over `t=0`, yet `shift=0` means centering over `T/2`
    shift_duration = shift * s
    tau = shift_duration

    # feed everything
    out = dft_windowed_sine(s, f0, T, tau, n_periods)
    return out


#%%###########################################################################
# Sandbox
# -------
# total, unwindowed number of samples
N_total = 512
# total, unwindowed duration
duration_total = 1
# windowed number of samples
N_windowed = 51
# frequency in Hertz
f0 = 50
# total time vector in seconds
t_total = np.linspace(0, duration_total, N_total, endpoint=0)
# shift in samples
shift = 21
# higher is more accurate
n_periods = 2000

# total sine, sampled
x_total = np.cos(2*np.pi * (f0 / duration_total) * t_total)
# windowed sine, sampled
x_windowed = x_total[shift:shift + N_windowed]

# formula
out0 = dft_windowed_sine_user(f0, shift, t_total, N_windowed, n_periods)
# user
out1 = fft(x_windowed)

# assert equality
assert np.allclose(out0, out1)
# confirm it's identical to the directly sampled (& using phase) expression
s = t_total[1] - t_total[0]
tau = shift * s
phi = 2*np.pi * f0 * tau
t = np.arange(N_windowed) * s
x_w_direct = np.cos(2*np.pi * f0 * t + phi)
assert np.allclose(x_windowed, x_w_direct)

#%%###########################################################################
# Testing
# -------
n_periods = 2000
for N_total in (256, 257):
    for duration_total in (1, 1.6):
        t_total = np.linspace(0, duration_total, N_total, endpoint=0)
        for N_windowed in (1, 2, 103, 104):
            for f0 in (0, 1, 2, 10, 51):
                x_total = np.cos(2*np.pi * (f0 / 1) * t_total)

                for shift in (0, 1, 2, 5, 10, 51):
                    # total sine, sampled
                    # windowed sine, sampled
                    x_windowed = x_total[shift:shift + N_windowed]

                    # formula
                    out0 = dft_windowed_sine_user(
                        f0, shift, t_total, N_windowed, n_periods)
                    # user
                    out1 = fft(x_windowed)

                    # assert equality
                    assert np.allclose(out0, out1), (
                        N_total, duration_total, N_windowed, f0, shift)
