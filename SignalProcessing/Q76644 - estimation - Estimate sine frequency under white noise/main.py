# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/76644/50076
# All code is written for readability, not performance.
import numpy as np
import matplotlib.pyplot as plt

# ensure the files can be found
import sys
from pathlib import Path
_dir = Path(__file__).parent
assert _dir.is_file() or _dir.is_dir(), str(_dir)
if not any(str(_dir).lower() == p.lower() for p in sys.path):
    sys.path.insert(0, str(_dir))

from estimators import est_freq, est_freq_multi, estimator_fns
from utils76644 import (
    make_x, run_test, print_progress,
    run_viz, get_viz_data, run_viz_multitone, get_viz_data_multitone,
    snr_db_amplitude_adjust, compute_crlbs)

print("Available estimators:\n  " + "\n  ".join(list(estimator_fns)))

#%% Configurations ###########################################################
# USER -----------------------------------------------------------------------
# prints test progress
VERBOSE = 1
# plot options
HEIGHT_SCALING = 1
WIDTH_SCALING = 1

# OTHER ----------------------------------------------------------------------
# execute some configs
w, h = 10*WIDTH_SCALING, 10*HEIGHT_SCALING
plt.rcParams['figure.figsize'] = [w, h]

# set certain defaults
f_N_all_nonints_large_N = (0.05393, 0.10696, 0.25494, 0.46595)
f_N_all_nonints_small_N = (0.053,   0.106,   0.254,   0.465)
f_N_all_ints_small_N    = (0.05,    0.10,    0.25,    0.46)
f_N_all_ints_large_N    = f_N_all_nonints_small_N
snrs_db_practical = np.linspace(-10, 50,  100)
snrs_db_wide      = np.linspace(100, 300, 50)

#%% Manual testing ###########################################################
np.random.seed(0)
name = 'cedron_3bin'
N = 10000
f = N*0.053123
phi = 1
x = np.cos(2*np.pi * f * np.arange(N)/N + phi)
# x = x + 1j*np.sin(2*np.pi * f * np.arange(N)/N)
x += np.random.randn(N) * 10

print(est_freq(x, name) / (f/N), sep='\n')

#%% Full testing #############################################################
# configure
seed = 0
sweep_mode = ('practical', 'wide')[0]
name0, name1 = 'cedron_3bin', 'kay_2'
# name0, name1 = 'cedron_3bin', 'cedron_2bin'
N = 100
n_trials = 2000
f_N_all = (f_N_all_nonints_small_N, f_N_all_nonints_large_N,
           f_N_all_ints_small_N,    f_N_all_ints_large_N)[0]

# execute some configs
if sweep_mode == 'practical':
    snrs = snrs_db_practical
else:
    snrs = snrs_db_wide
crlbs = compute_crlbs(N, snrs, T=1)

# run test
errs0_all, errs1_all = {}, {}
for f_N in f_N_all:
    f = f_N * N
    errs0, errs1 = run_test(f, N, n_trials, name0, name1, snrs, seed)
    errs0_all[f_N] = errs0
    errs1_all[f_N] = errs1

    if VERBOSE:
        print_progress(f_N, N, n_trials, name0, name1, f_N_all)

#%% Visualize ################################################################
plot_data = {}
for f_N in f_N_all:
    plot_data[f_N] = get_viz_data(errs0_all[f_N], errs1_all[f_N])
ymin = int(np.floor(np.min(np.array(list(plot_data.values()))[:, 1:])))

fig, axes = plt.subplots(2, 2, figsize=(w*1.6, h*1.6), layout='constrained')

for i, f_N in enumerate(f_N_all):
    legend2 = bool(i == 0)
    ylabel = bool(i % 2 == 0)
    run_viz(*plot_data[f_N], f_N, N, n_trials, snrs, crlbs, ymin=ymin,
            figax=(fig, axes.flat[i]), legend2=legend2, ylabel=ylabel)
plt.show()

#%% "Extreme" example ########################################################
N = 100000
f_N = .053056
snr = -30
f = f_N * N

errs = []
for _ in range(2000):
    x = make_x(N, f, snr)
    f_est = est_freq(x, 'cedron_3bin')
    errs.append((f_N - f_est)**2)
mse = np.mean(errs)

print(mse)

#%%
x, xo = make_x(N, f, snr, get_xo=True)
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5), layout='constrained')
axes[0].plot(xo[:500])
axes[0].set_title("Original signal, zoomed | N=100000",
                  weight='bold', loc='left', fontsize=24)
axes[1].plot(x[:500])
axes[1].set_title("Noisy, zoomed | SNR=1/1000 (-30dB)",
                  weight='bold', loc='left', fontsize=24)

#%% Multi-tone Example #######################################################
N = 10000
n_trials = 2000
# include integer case
f_N_all = (0.05305, 0.10605, 0.254, 0.46505)
# f_N_all = (0.10601, 0.10644, 0.10696, 0.10747)
A_all = (0.5, 0.8, 1.2, 1.5)  # mean=1
# each `A` will have different SNR, so extend the range so we can plot all
# under a common snr
snrs_bounds = (snrs_db_practical[0], snrs_db_practical[-1])
snrs = np.linspace(snrs_bounds[0] - 10, snrs_bounds[1] + 15,
                   int(len(snrs_db_practical)*1.25))
n_tones = len(f_N_all)

# compute SNRs for later
snrs_all = {f_N: snr_db_amplitude_adjust(snrs, A)
            for f_N, A in zip(f_N_all, A_all)}

np.random.seed(seed)
_base_arg = 2*np.pi*np.arange(N)/N
errs_all0, errs_all1 = [{f_N: {snr: [] for snr in snrs} for f_N in f_N_all}
                        for _ in range(2)]
for snr in snrs:
    noise_var = 0.5 / 10**(snr/10)  # unit-amplitude case
    noise_std = np.sqrt(noise_var)

    for _ in range(n_trials):
        x = np.random.randn(N) * noise_std
        for f_N, A in zip(f_N_all, A_all):
            phi = np.random.uniform(0, 1) * (2*np.pi)
            x += A * np.cos(_base_arg*(f_N * N) + phi)

        f_ests0, f_ests1 = est_freq_multi(x, n_tones=n_tones)
        for f_N in f_N_all:
            errs_all0[f_N][snr].append(np.min((f_ests0 - f_N)**2))
            errs_all1[f_N][snr].append(np.min((f_ests1 - f_N)**2))

    for f_N in f_N_all:
        errs_all0[f_N][snr] = (
            np.mean(errs_all0[f_N][snr]), np.std(errs_all0[f_N][snr]))
        errs_all1[f_N][snr] = (
            np.mean(errs_all1[f_N][snr]), np.std(errs_all1[f_N][snr]))
    if VERBOSE:
        print(end='.')

#%% Visualize ################################################################
plot_data = {}
for f_N in f_N_all:
    crlbs = compute_crlbs(N, snrs_all[f_N], T=1)
    plot_data[f_N] = get_viz_data_multitone(errs_all0[f_N], errs_all1[f_N],
                                            snrs_all[f_N], crlbs, snrs_bounds)

# exclude `a` and zero stdev (log -inf)
vals = np.array([np.array(list(plot_data.values()))[:, i] for i in
                 (1, 3, 5)])
# take log of crlbs
vals[-1] = np.log10(vals[-1])
vals[np.isinf(vals)] = 0
ymin = int(np.floor(np.min(vals)))

fig, axes = plt.subplots(2, 2, figsize=(w*1.6, h*1.6), layout='constrained')

for i, (f_N, A) in enumerate(zip(f_N_all, A_all)):
    legend2 = bool(i == 0)
    ylabel = bool(i % 2 == 0)
    run_viz_multitone(*plot_data[f_N], f_N, A, snrs_bounds, ymin=ymin,
                      figax=(fig, axes.flat[i]), legend2=legend2, ylabel=ylabel)
fig.suptitle("Multi-tone Estimation: N={}, n_trials={}".format(N, n_trials),
             weight='bold', fontsize=28)
plt.show()
