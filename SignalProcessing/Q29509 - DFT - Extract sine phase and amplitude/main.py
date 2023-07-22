# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/29509/50076
import warnings
import numpy as np
import matplotlib.pyplot as plt

# ensure the files can be found
import sys
from pathlib import Path
_dir = Path(__file__).parent
assert _dir.is_file() or _dir.is_dir(), str(_dir)
if not any(str(_dir).lower() == p.lower() for p in sys.path):
    sys.path.insert(0, str(_dir))

from estimators import est_amp_phase_cedron_2bin, est_f_cedron_2bin

#%% Helpers ##################################################################
def cisoid(N, f, phi=0):
    return (np.cos(2*np.pi*f*np.arange(N)/N + phi) +
            np.sin(2*np.pi*f*np.arange(N)/N + phi)*1j)

def est_and_append_errs(x, f_N, A, phi, errs_A_alt, errs_phi,
                        errs_A=None, errs_f=None):
    if errs_f is not None:
        f_est = est_f_cedron_2bin(x)

    A_est, phi_est = est_amp_phase_cedron_2bin(x)
    err_A, err_A_alt, err_phi = get_errs_A_phi(A_est, phi_est, A, phi)

    if errs_A is not None:
        errs_A.append(err_A)
    if errs_f is not None:
        errs_f.append((f_est - f_N)**2)

    # normalized
    errs_A_alt.append(err_A_alt)

    errs_phi.append(err_phi)


def get_errs_A_phi(A_est, phi_est, A, phi):
    err_A = (A_est - A)**2
    err_A_alt = (1 - A_est/A)**2

    # note inherent ambiguity; discard sign in this case
    if abs(abs(phi_est) - np.pi) < .001:
        err_phi = (abs(phi_est) - abs(phi))**2
    else:
        err_phi = (phi_est - phi)**2
    return err_A, err_A_alt, err_phi


#%% Manual testing ###########################################################
N = 3
f = 0.0035035*N
phi = -1
A = 1

x = A*cisoid(N, f, phi).real
f_N = f/N

f_est = est_f_cedron_2bin(x)
A_est, phi_est = est_amp_phase_cedron_2bin(x)

print(f_est / f_N)
print(A_est / A)
print(phi_est / phi)

#%% Test full (noiseless) ####################################################
N = 3

errs_f_fmax, errs_A_fmax, errs_phi_fmax = [], [], []
errs_A_alt_fmax = []
# 0.5 takes more coding but can be handled

# make freqs
f_N_all = []
for f_N in np.linspace(0, 0.5, 203, endpoint=False):
    # skip near-integer frequency per numeric instability
    # (robust version not implemented)
    if (f_N * N) % 1 < .01:
        continue
    f_N_all.append(f_N)
f_N_all = np.array(f_N_all)

# make amplitudes
A_all = np.logspace(np.log10(1e-3), np.log10(1000), 100)
# make phases
phi_all = np.linspace(-np.pi, np.pi, 100)

for f_N in f_N_all:
    errs_f, errs_A, errs_phi = [], [], []
    errs_A_alt = []
    f = f_N * N

    for A in A_all:
        for phi in phi_all:
            x = A*cisoid(N, f, phi).real
            est_and_append_errs(x, f_N, A, phi, errs_A_alt, errs_phi,
                                errs_A, errs_f)

    errs_f_fmax.append(np.max(errs_f))
    errs_A_fmax.append(np.max(errs_A))
    errs_phi_fmax.append(np.max(errs_phi))
    errs_A_alt_fmax.append(np.max(errs_A_alt))

#%% Visualize ################################################################
fig, ax = plt.subplots(layout='constrained', figsize=(12, 8))

ax.plot(f_N_all, np.log10(errs_f_fmax))
ax.plot(f_N_all, np.log10(errs_A_alt_fmax))
ax.plot(f_N_all, np.log10(errs_phi_fmax))

title = ("Max errors: frequency, amplitude, phase | N={}\n"
         "n_freqs, n_amps, n_phases = {}, {}, {}"
         ).format(N, len(f_N_all), len(A_all), len(phi_all))

ax.set_title(title, weight='bold', fontsize=24, loc='left')
ax.set_ylabel("Squared Error (log10)", fontsize=22)
ax.set_xlabel("f/N", fontsize=22)
ax.legend(["f", "A", "phi"], fontsize=22)

plt.show()

#%% Test full (noisy) ########################################################
np.random.seed(0)
N = 300
n_trials = 50

snrs = np.linspace(-10, 50, 25)

# make freqs
f_N_all = []
for f_N in np.linspace(1/N, 0.5-1/N, 42):
    # skip near-integer frequency per numeric instability
    # (robust version not implemented)
    if (f_N * N) % 1 < .01:
        continue
    f_N_all.append(f_N)
f_N_all = np.array(f_N_all)

# make amplitudes
A_all = np.logspace(np.log10(1e-3), np.log10(1000), 10)

errs_A_alt_all, errs_phi_all = [], []
for snr in snrs:
    errs_A_alt, errs_phi = [], []
    for f_N in f_N_all:
        f = f_N * N
        for A in A_all:
            for _ in range(n_trials):
                phi = np.random.uniform(-np.pi, np.pi)
                xo = A * cisoid(N, f, phi).real
                noise_var = xo.var() / 10**(snr/10)
                noise = np.random.randn(N) * np.sqrt(noise_var)
                x = xo + noise

                with warnings.catch_warnings(record=True) as ws:
                    est_and_append_errs(x, f_N, A, phi, errs_A_alt, errs_phi)
                    # if len(ws) > 0:
                    #     print(snr, f_N, f_N*N, A, phi, sep='\n')
                    #     1/0
    errs_A_alt_all.append(np.mean(errs_A_alt))
    errs_phi_all.append(np.mean(errs_phi))
    print(end='.', flush=True)

#%% Visualize ################################################################
fig, ax = plt.subplots(layout='constrained', figsize=(12, 8))

ax.plot(snrs, np.log10(errs_A_alt_all))
ax.plot(snrs, np.log10(errs_phi_all))

title = ("N={}, f/N=lin sweep, phase=randu(-pi, pi)\n"
         "n_freqs={}, n_amps={}, n_trials_per_f_and_A={}"
         ).format(N, len(f_N_all), len(A_all), n_trials)

ax.set_title(title, weight='bold', fontsize=24, loc='left')
ax.set_ylabel("MSE (log10)", fontsize=22)
ax.set_xlabel("SNR [dB]", fontsize=22)
ax.legend(["A", "phi"], fontsize=22)
ax.set_ylim(-10, 1)

plt.show()

#%% OP's case ################################################################
np.random.seed(0)
A = 1
f = 30.1
N = 300
t = np.linspace(0, 1, N, endpoint=False)
n_trials = 10000

errs_A_alt, errs_phi = [], []
for _ in range(n_trials):
    phi = np.random.uniform(-np.pi, np.pi)
    xo = A*np.cos(2*np.pi*f*t + phi)
    noise = np.random.normal(0, 0.05, len(t))
    x = xo + noise

    A_est, phi_est = est_amp_phase_cedron_2bin(x)

    _, err_A_alt, err_phi = get_errs_A_phi(A_est, phi_est, A, phi)
    errs_A_alt.append(err_A_alt)
    errs_phi.append(err_phi)

SNR = 10*np.log10((A/2)/.05**2)
print("MSE (log10): A={:.3g}, phi={:.3g} -- SNR={:.3g}, N={}, n_trials={}".format(
    np.mean(errs_A_alt), np.mean(errs_phi), SNR, N, n_trials))
