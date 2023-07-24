# -*- coding: utf-8 -*-
"""Testing, visualization, and utility functions."""
# https://dsp.stackexchange.com/q/76644/50076
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from estimators import est_freq, est_freq_multi
from timeit import Timer
from functools import partial

# USER CONFIGS
# plot options
HEIGHT_SCALING = 1
WIDTH_SCALING = 1

# OTHER ----------------------------------------------------------------------
# execute some configs
w, h = 10*WIDTH_SCALING, 10*HEIGHT_SCALING
plt.rcParams['figure.figsize'] = [w, h]

# reusables
snrs_db_practical = np.linspace(-10, 50,  100)
snrs_db_wide      = np.linspace(100, 300, 50)

# Testing ####################################################################
def make_x(N, f, snr=None, _base_arg=None, get_xo=False, real=True):
    if _base_arg is None:
        _base_arg = 2*np.pi*f*np.arange(N)/N

    phi = np.random.uniform(0, 1) * (2*np.pi)
    if real:
        xo = np.cos(_base_arg + phi)
    else:
        xo = np.cos(_base_arg + phi) + 1j*np.sin(_base_arg + phi)

    if snr is not None:
        noise_var = xo.var() / 10**(snr/10)
        if real:
            noise = randn(N) * np.sqrt(noise_var)
        else:
            noise = crandn(N) * np.sqrt(noise_var)
    else:
        noise = 0
    x = xo + noise
    return x if not get_xo else (x, xo)


def run_test(f_N_all, N, n_trials, name0, name1, real=True, seed=0,
             sweep_mode='practical', snrs=None, verbose=True):
    # execute some configs
    if snrs is None:
        if sweep_mode == 'practical':
            snrs = snrs_db_practical
        else:
            snrs = snrs_db_wide

    # run test
    errs0_all, errs1_all = {}, {}
    for f_N in f_N_all:
        f = f_N * N
        errs0, errs1 = _run_test(f, N, n_trials, name0, name1, snrs, real, seed)
        errs0_all[f_N] = errs0
        errs1_all[f_N] = errs1

        if verbose:
            print_progress(f_N, N, n_trials, name0, name1, f_N_all)

    crlbs = compute_crlbs(N, snrs, T=1)
    return errs0_all, errs1_all, snrs, crlbs


def _run_test(f, N, n_trials, name0, name1, snrs, real, seed):
    np.random.seed(seed)
    _base_arg = 2*np.pi*f*np.arange(N)/N

    errs0, errs1 = {}, {}
    for snr in snrs:
        errs0[snr], errs1[snr] = [], []

        for _ in range(n_trials):
            x = make_x(N, f, snr, _base_arg, real=real)

            f_est0, f_est1 = est_freq(x, names=(name0, name1), real=real)
            err0 = (f_est0 - f/N)**2
            err1 = (f_est1 - f/N)**2
            errs0[snr].append(err0)
            errs1[snr].append(err1)

        errs0[snr] = (np.mean(errs0[snr]), np.std(errs0[snr]))
        errs1[snr] = (np.mean(errs1[snr]), np.std(errs1[snr]))

    return errs0, errs1


def run_test_multitone(f_N_all, A_all, N, n_trials, name0, name1, snrs, seed=0,
                       verbose=True):
    np.random.seed(seed)
    n_tones = len(f_N_all)
    _base_arg = 2*np.pi*np.arange(N)/N

    errs0_all, errs1_all = [{f_N: {snr: [] for snr in snrs} for f_N in f_N_all}
                            for _ in range(2)]
    for snr in snrs:
        noise_var = 0.5 / 10**(snr/10)  # unit-amplitude case
        noise_std = np.sqrt(noise_var)

        for _ in range(n_trials):
            x = np.random.randn(N) * noise_std
            for f_N, A in zip(f_N_all, A_all):
                phi = np.random.uniform(0, 1) * (2*np.pi)
                x += A * np.cos(_base_arg*(f_N * N) + phi)

            f_ests0, f_ests1 = est_freq_multi(x, names=(name0, name1),
                                              n_tones=n_tones)
            for f_N in f_N_all:
                errs0_all[f_N][snr].append(np.min((f_ests0 - f_N)**2))
                errs1_all[f_N][snr].append(np.min((f_ests1 - f_N)**2))

        for f_N in f_N_all:
            errs0_all[f_N][snr] = (
                np.mean(errs0_all[f_N][snr]), np.std(errs0_all[f_N][snr]))
            errs1_all[f_N][snr] = (
                np.mean(errs1_all[f_N][snr]), np.std(errs1_all[f_N][snr]))
        if verbose:
            print(end='.')

    crlbs = compute_crlbs(N, snrs, T=1)
    # compute SNRs for later
    snrs_f_N = {f_N: snr_db_amplitude_adjust(snrs, A)
                for f_N, A in zip(f_N_all, A_all)}
    return errs0_all, errs1_all, snrs_f_N, crlbs


def print_progress(f_N, N, n_trials, name0, name1, f_N_all):
    longest = max(len(str(f_N)) for f_N in f_N_all)
    fmt = "f={:<" + str(longest) + ".6g}"
    txt = (fmt + " done | N, n_trials, name0, name1 = {}, {}, {}, {}"
           ).format(f_N, N, n_trials, name0, name1)
    print(txt, flush=True)

# Visualization ##############################################################
def run_viz(errs0_all, errs1_all, snrs, crlbs,
            f_N_all, N, n_trials, names=("Cedron", "Kay_2")):
    viz_names = names
    plot_data = {}
    for f_N in f_N_all:
        plot_data[f_N] = get_viz_data(errs0_all[f_N], errs1_all[f_N])
    ymin = int(np.floor(np.min(np.array(list(plot_data.values()))[:, 1:])))

    fig, axes = plt.subplots(2, 2, figsize=(w*1.6, h*1.6), layout='constrained')

    for i, f_N in enumerate(f_N_all):
        legend2 = bool(i == 0)
        ylabel = bool(i % 2 == 0)
        _run_viz(*plot_data[f_N], f_N, N, n_trials, snrs, crlbs, ymin=ymin,
                 figax=(fig, axes.flat[i]), legend2=legend2, ylabel=ylabel,
                 viz_names=viz_names)
    plt.show()


def _run_viz(a, b0mn, b1mn, b0sd, b1sd, f_N, N, n_trials, snrs, crlbs,
             ymin=None, legend2=False, figax=None, ylabel=True,
             viz_names=("Cedron", "Kay_2")):
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
    title = "f/N = {:.6g}, N={}, n_trials={}".format(f_N, N, n_trials)
    _basic_style(ax, ymin, snrs, title, N, n_trials, ylabel=ylabel)

    # legends
    legend = list(viz_names)
    if crlbs is not None:
        legend = ["CRLB"] + legend
    _legend2(ax, legend)


def run_viz2(errs0_all, errs1_all, snrs, crlbs,
             f_N_all, N, n_trials, names=("Cedron", "Kay_2")):
    viz_names = names
    plot_data = get_viz_data2(errs0_all, errs1_all)
    n_freqs = len(f_N_all)

    ymin = int(np.floor(min(np.min(crlbs), np.min(np.array(plot_data[1:])))))
    fig, ax = plt.subplots(figsize=(w*1, h*1), layout='constrained')
    _run_viz2(*plot_data, N, n_freqs, n_trials, snrs, crlbs, ymin=ymin,
              figax=(fig, ax), viz_names=viz_names)
    plt.show()


def _run_viz2(a, b0mn, b1mn, b0sd, b1sd, N, n_freqs, n_trials, snrs, crlbs,
              ymin=None, figax=None, viz_names=("Cedron", "DFT_quadratic")):
    # plot
    if figax is None:
        fig, ax = plt.subplots(layout='constrained')
    else:
        fig, ax = figax

    color2 = ('tab:green' if 'kay' in viz_names[1].lower() else
              'tab:cyan')

    if crlbs is not None:
        ax.plot(a, np.log10(crlbs),      linewidth=3)
    ax.plot(a, b0mn, color='tab:orange', linewidth=3)
    ax.plot(a, b1mn, color=color2,       linewidth=3)
    ax.plot(a, b0sd, color='tab:orange', linewidth=2, linestyle='--')
    ax.plot(a, b1sd, color=color2,       linewidth=2, linestyle='--')

    # configure axes, set title
    extra = (" Dq_Npad=2048, " if viz_names[1] == 'DFT_quadratic' else
             "")
    title = ("N={},{} f/N=lin sweep\n"
             "n_freqs={}, n_trials_per_freq={}").format(
                 N, extra, n_freqs, n_trials)
    _basic_style(ax, ymin, snrs, title, N, n_trials, ylabel=True)

    # legends
    legend = ["CRLB", *viz_names]
    _legend2(ax, legend)


def _legend2(ax, legend):
    first_legend = ax.legend(legend, fontsize=22, loc=1)
    lg0 = mlines.Line2D([], [], color='k', linewidth=3, label='mean')
    lg1 = mlines.Line2D([], [], color='k', linewidth=3, label='std',
                        linestyle='--')
    ax.add_artist(first_legend)
    ax.legend(handles=[lg0, lg1], loc='lower left', fontsize=22)


def _basic_style(ax, ymin, snrs, title, N, n_trials, ylabel=True):
    ax.set_ylim(ymin, 0)
    ax.set_xlim(np.min(snrs), np.max(snrs))
    ax.set_xlabel("SNR [dB]", size=20)
    if ylabel:
        ax.set_ylabel("MSE (log10)", size=20)
    ax.set_title(title, weight='bold', fontsize=24)


def get_viz_data(*errs_all):
    a = np.array(list(errs_all[0]))

    # retrieve means & SDs, make log
    bmns, bsds = [], []
    for errs in errs_all:
        b = np.log10(np.array(list(errs.values())))
        bmn, bsd = b[:, 0], b[:, 1]
        bmns.append(bmn)
        bsds.append(bsd)
    return a, *bmns, *bsds


def get_viz_data2(*errs_all):
    f_N_all = list(errs_all[0])
    snrs = list(errs_all[0][f_N_all[0]])
    a = snrs

    # restructure
    # element of `errs_all` is structured
    #     {f_N: {snr: (float, float)}}
    # convert to
    #     {snr: [(float, float)]}
    errs_all_re = []
    for errs in errs_all:
        errs_re = {f_N: np.array(list(errs[f_N].values())) for f_N in f_N_all}
        errs_all_re.append(errs_re)

    # retrieve means & SDs, make log
    bmns_all, bsds_all = [], []
    for errs_re in errs_all_re:
        b = np.array(list(errs_re.values()))
        bmns, bsds = b[..., 0], b[..., 1]
        # average along frequency; important to take log first, else
        # result's dominated by outliers
        bmn, bsd = [np.mean(np.log10(g), axis=0) for g in (bmns, bsds)]
        bmns_all.append(bmn)
        bsds_all.append(bsd)

    bmns_all, bsds_all = [np.array(g) for g in (bmns_all, bsds_all)]
    return a, *bmns_all, *bsds_all


def run_viz_multitone(errs0_all, errs1_all, snrs_f_N, crlbs,
                      f_N_all, A_all, N, n_trials, snrs_bounds,
                      names=("Cedron", "DFT_argmax")):
    viz_names = names
    plot_data = {}
    for f_N in f_N_all:
        crlbs = compute_crlbs(N, snrs_f_N[f_N], T=1)
        plot_data[f_N] = get_viz_data_multitone(errs0_all[f_N], errs1_all[f_N],
                                                snrs_f_N[f_N], crlbs, snrs_bounds)

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
        _run_viz_multitone(*plot_data[f_N], f_N, A, snrs_bounds, ymin=ymin,
                           figax=(fig, axes.flat[i]), legend2=legend2,
                           ylabel=ylabel, viz_names=viz_names)
    fig.suptitle("Multi-tone Estimation: N={}, n_trials={}".format(N, n_trials),
                 weight='bold', fontsize=28)
    plt.show()


def _run_viz_multitone(a, b0mn, b1mn, b0sd, b1sd, c, f_N, A, snrs_bounds,
                       ymin=None, legend2=False, figax=None, ylabel=True,
                       viz_names=("Cedron", "DFT_argmax")):
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
    legend = ["CRLB-single", *viz_names]
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


def randn(N):
    return np.random.randn(N)


def crandn(N):
    return np.random.randn(N) + 1j*np.random.randn(N)


def cisoid(N, f, phi=0):
    return (np.cos(2*np.pi*f*np.arange(N)/N + phi) +
            np.sin(2*np.pi*f*np.arange(N)/N + phi)*1j)

# Benchmarking ###############################################################
def timeit(fn, args, n_trials=200, repeats=10):
    """
    https://stackoverflow.com/a/8220943/10133797
    https://stackoverflow.com/a/24105845/10133797
    """
    return min(Timer(partial(fn, args)).repeat(repeats, n_trials)) / n_trials
