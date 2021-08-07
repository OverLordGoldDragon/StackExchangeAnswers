import numpy as np
import matplotlib.pyplot as plt


def plot(x, title):
    fig = plt.figure()
    plt.plot(np.abs(x))
    plt.scatter(np.arange(len(x)), np.abs(x), s=10)
    plt.title(title, weight='bold', fontsize=18, loc='left')
    return fig

def plot_T(x, Tmax):
    if Tmax == 0:
        title = "|H(w)|: x(n)"
    elif Tmax == 1:
        title = "|H(w)|: x(n) - x(n - 1)"
    elif Tmax == 2:
        title = "|H(w)|: x(n) - x(n - 1) + x(n - 2)"
    else:
        title = "|H(w)|: x(n) - x(n - 1) + x(n - 2) - ... x(n - %s)" % Tmax

    fig = plot(x, title, scatter=1)
    plt.ylim(-.05, 1.05)

    plt.savefig(f'im{Tmax}.png', bbox_inches='tight')
    plt.close(fig)

def csoid(f):
    return (np.cos(2*np.pi* f * t) -
            np.sin(2*np.pi* f * t) * 1j)

#%%# Direct frequency response ###############################################
N = 32
t = np.linspace(0, 1, N, 0)

for Tmax in range(N):
    x = np.sum([(-1)**T * csoid(T) for T in range(Tmax + 1)], axis=0)
    x /= np.abs(x).max()
    plot_T(x, Tmax)

#%%# WGN example #############################################################
def plot_and_save(x, title, savepath):
    fig = plot(x, title)
    plt.savefig(savepath, bbox_inches='tight')
    plt.close(fig)

np.random.seed(69)
x = np.random.randn(32)
xf0 = np.fft.fft(x)
x = x - np.roll(x, 1) + np.roll(x, 2)
xf1 = np.fft.fft(x)

plot_and_save(xf0, "|X(w)|: x(n)", "WGN0.png")
plot_and_save(xf1, "|X(w)|: x(n) - x(n - 1) + x(n - 2)", "WGN1.png")
