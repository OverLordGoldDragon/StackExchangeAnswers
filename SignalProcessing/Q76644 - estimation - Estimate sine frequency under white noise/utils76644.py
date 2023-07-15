# -*- coding: utf-8 -*-
"""Testing, visualization, and utility functions."""
# https://dsp.stackexchange.com/q/76644/50076
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from estimators import est_freq

# Testing ####################################################################
def make_x(N, f, snr, _base_arg=None, get_xo=False):
    if _base_arg is None:
        _base_arg = 2*np.pi*f*np.arange(N)/N
    phi = np.random.uniform(0, 1) * (2*np.pi)
    xo = np.cos(_base_arg + phi)
    noise_var = xo.var() / 10**(snr/10)
    noise = np.random.randn(N) * np.sqrt(noise_var)
    x = xo + noise
    return x if not get_xo else (x, xo)


def run_test(f, N, n_trials, name0, name1, snrs, seed):
    np.random.seed(seed)
    _base_arg = 2*np.pi*f*np.arange(N)/N

    errs0, errs1 = {}, {}
    for snr in snrs:
        errs0[snr], errs1[snr] = [], []

        for _ in range(n_trials):
            x = make_x(N, f, snr, _base_arg)

            f_est0 = est_freq(x, name0)
            f_est1 = est_freq(x, name1)
            err0 = (f_est0 - f/N)**2
            err1 = (f_est1 - f/N)**2
            errs0[snr].append(err0)
            errs1[snr].append(err1)

        errs0[snr] = (np.mean(errs0[snr]), np.std(errs0[snr]))
        errs1[snr] = (np.mean(errs1[snr]), np.std(errs1[snr]))

    return errs0, errs1


def print_progress(f_N, N, n_trials, name0, name1, f_N_all):
    longest = max(len(str(f_N)) for f_N in f_N_all)
    fmt = "f={:<" + str(longest) + ".6g}"
    txt = (fmt + " done | N, n_trials, name0, name1 = {}, {}, {}, {}"
           ).format(f_N, N, n_trials, name0, name1)
    print(txt, flush=True)

# Visualization ##############################################################
def run_viz(a, b0mn, b1mn, b0sd, b1sd, f_N, N, n_trials, snrs, crlbs,
            ymin=None, legend2=False, figax=None, ylabel=True):
    # plot
    if figax is None:
        fig, ax = plt.subplots(layout='constrained')
    else:
        fig, ax = figax

    if crlbs is not None:
        ax.plot(a, np.log10(crlbs), linewidth=3)
    ax.plot(a, b0mn, color='tab:orange', linewidth=3)
    ax.plot(a, b1mn, color='tab:green',  linewidth=3)
    ax.plot(a, b0sd, color='tab:orange', linewidth=2, linestyle='--')
    ax.plot(a, b1sd, color='tab:green',  linewidth=2, linestyle='--')

    # configure axes, set title
    ax.set_ylim(ymin, 0)
    ax.set_xlim(snrs.min(), snrs.max())
    ax.set_xlabel("SNR [dB]", size=20)
    if ylabel:
        ax.set_ylabel("MSE (log10)", size=20)
    ax.set_title("f/N = {:.6g}, N={}, n_trials={}".format(f_N, N, n_trials),
                 weight='bold', fontsize=24)

    # legends
    legend = ["Cedron", "Kay 2"]
    if crlbs is not None:
        legend = ["CRLB"] + legend
    first_legend = ax.legend(legend, fontsize=22, loc=1)
    if legend2:
        lg0 = mlines.Line2D([], [], color='k', linewidth=3, label='mean')
        lg1 = mlines.Line2D([], [], color='k', linewidth=3, label='std',
                            linestyle='--')
        ax.add_artist(first_legend)
        ax.legend(handles=[lg0, lg1], loc='lower left', fontsize=22)


def get_viz_data(errs0, errs1):
    # retrieve means & SDs, make log
    a = np.array(list(errs0))
    b0 = np.log10(np.array(list(errs0.values())))
    b1 = np.log10(np.array(list(errs1.values())))
    b0mn, b1mn = b0[:, 0], b1[:, 0]
    b0sd, b1sd = b0[:, 1], b1[:, 1]
    return a, b0mn, b1mn, b0sd, b1sd


def run_viz_multitone(a, b0mn, b1mn, b0sd, b1sd, c, f_N, A, snrs_bounds,
                      ymin=None, legend2=False, figax=None, ylabel=True):
    # plot
    fig, ax = figax

    ax.plot(a, np.log10(c), linewidth=3)
    ax.plot(a, b0mn, color='tab:orange', linewidth=3)
    ax.plot(a, b1mn, color='tab:cyan',   linewidth=3)
    ax.plot(a, b0sd, color='tab:orange', linewidth=2, linestyle='--')

    # configure axes, set title
    ax.set_ylim(ymin, 0)
    ax.set_xlim(*snrs_bounds)
    ax.set_xlabel("SNR [dB]", size=20)
    if ylabel:
        ax.set_ylabel("log10(MSE)", size=20)
    ax.set_title("f/N = {:.6g}, A={}".format(f_N, A),
                 weight='bold', fontsize=24)

    # legends
    legend = ["CRLB-single", "Cedron", "DFT_argmax"]
    first_legend = ax.legend(legend, fontsize=22, loc=1)
    if legend2:
        lg0 = mlines.Line2D([], [], color='k', linewidth=3, label='mean')
        lg1 = mlines.Line2D([], [], color='k', linewidth=3, label='std',
                            linestyle='--')
        ax.add_artist(first_legend)
        ax.legend(handles=[lg0, lg1], loc='lower left', fontsize=22)


def get_viz_data_multitone(errs0, errs1, snrs, crlbs, snrs_bounds):
    # retrieve means & SDs, make log
    a = list(snrs)
    b0 = list(np.log10(np.array(list(errs0.values()))))
    b1 = list(np.log10(np.array(list(errs1.values()))))
    c = list(crlbs)
    # exclude points outside `snrs_bounds`
    for snr in snrs:
        if snr < snrs_bounds[0]:
            for ls in (a, b0, b1, c):
                ls.pop(0)
        elif snr > snrs_bounds[1]:
            for ls in (a, b0, b1, c):
                ls.pop(-1)
    a, b0, b1, c = [np.array(g) for g in (a, b0, b1, c)]
    b0mn, b1mn = b0[:, 0], b1[:, 0]
    b0sd, b1sd = b0[:, 1], b1[:, 1]
    return a, b0mn, b1mn, b0sd, b1sd, c

# Misc #######################################################################
def compute_crlbs(N, snr_db, T=1):
    """Cramer-Rao Lower Bound, all unknown, unbiased estimator
        N: number of samples,
        snr_db: Signal/Noise ratio in dB
        T: duration in sec
    https://www.mdpi.com/1424-8220/13/5/5649 , Eq 48
    """
    snr = 10**(snr_db/10)
    crlbs = 12 / ((2*np.pi)**2 * snr * T**2 * N * (N**2 - 1))
    return crlbs


def snr_db_amplitude_adjust(snr_db, A):
    """
    # General:
    snr_db = 10 * log10( (A**2/2) / (sigma**2) )

    # Consider:
    snr_db0 = 10 * log10( (1/2)    / (sigma**2) )
    snr_db1 = 10 * log10( (A**2/2) / (sigma**2) )

    # Got: snr_db0. Want: snr_db1.
    [10**(snr_db0/10)] = [(1/2) / sigma**2] * A**2
    [10**(snr_db0/10)] * A**2 = [(1/2) / sigma**2] * A**2
    G = 10**(snr_db0/10) * A**2
    10 * log10(G) = 10 * log10((A**2/2) / sigma**2)
    10 * log10(G) = snr_db1
    """
    return 10 * np.log10(10**(snr_db/10) * A**2)
