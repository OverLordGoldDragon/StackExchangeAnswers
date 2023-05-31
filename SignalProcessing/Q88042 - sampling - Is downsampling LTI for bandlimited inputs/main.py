# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/88042/50076
import numpy as np
from numpy.fft import fft, ifft
# from .toolkit import fft_upsample

# NOTE: linearity isn't tested!

#%%###########################################################################
# Helpers
# -------
def crandn(N):
    return np.random.randn(N) + 1j*np.random.randn(N)

# note, this `op` is deliberately incomplete relative to the answer's formulation
# (i.e. omits `fft_upsample`, so that we can test non-upsampled cases)
def op(a, b, d, x_only):
    if x_only:
        return ifft(fft(a[::d]))
    else:
        return ifft(fft(a[::d]) * fft(b[::d] * d))

def div(a, b):
    return int(np.ceil(a / b))

#%%###########################################################################
# Testing
# -------
optional_successes = {i: [] for i in range(5)}

for real in (True, False):
  for x_bandlimited in (True, False):
    for h_bandlimited in (True, False):
      for x_only in (True, False):
        if x_only and not h_bandlimited:
            # duplicate case
            continue
        for N in (16, 40, 128):
          for d in range(2, N//2, 2):
            # d==N//2 is an edge case that can be handled but no need
            for s in range(1, N):
              # prepare --------------------------------------------------------
              # check if loop is valid
              if np.ceil(N / d) % 2 != 0 or not np.log2(d).is_integer():
                  # `fft_upsample` doesn't handle this case, nor does it seem
                  # that many of such cases can be handled
                  continue

              # store reusables
              both_bandlimited = bool(x_bandlimited and h_bandlimited)
              one_bandlimited = bool(x_bandlimited or h_bandlimited)
              not_bandlimited = bool(not x_bandlimited and not h_bandlimited)

              # generate filter, signal
              x, h = crandn(N), crandn(N)
              if real:
                  x, h = x.real, h.real
              if x_bandlimited:
                  xf = fft(x)
                  xf[N//d//2:-(N//d//2 - 1)] = 0
                  x = ifft(xf)
              if h_bandlimited:
                  hf = fft(h)
                  hf[N//d//2:-(N//d//2 - 1)] = 0
                  h = ifft(hf)
              if real:
                  assert np.allclose(max(abs(x.imag)), 0)
                  assert np.allclose(max(abs(h.imag)), 0)
                  x, h = x.real, h.real

              # execute --------------------------------------------------------
              # get outputs
              o0a = op(np.roll(x, 0), h, 1, x_only)
              o0b = op(np.roll(x, s), h, 1, x_only)
              o1a = op(np.roll(x, 0), h, d, x_only)
              o1b = op(np.roll(x, s), h, d, x_only)

              # get upsampled outputs
              o1au, o1bu = [fft_upsample(o, d, time_to_time=True)
                            for o in (o1a, o1b)]

              # validate -------------------------------------------------------
              # prepare info in case assert fails
              cfg = dict(real=real,
                         x_bandlimited=x_bandlimited,
                         h_bandlimited=h_bandlimited,
                         x_only=x_only, N=N, d=d, s=s)
              info = "\n  " + "\n  ".join(f"{k}={v}" for k, v in cfg.items())
              _err = lambda i: f'{i}:{info}'

              # fetch expectations
              cond = {
                  0: True,
                  1: False,
                  2: (s / d).is_integer(),
                  3: ((x_only and (x_bandlimited or (s / d).is_integer())) or
                      (one_bandlimited and (s / d).is_integer())),
                  4: both_bandlimited,
              }
              optional = {
                  0: False,
                  1: True,
                  # only integer `s/d` must pass
                  2: False if cond[2] else True,
                  # only bandlimited or integer `s/d` must pass
                  3: False if cond[3] else True,
                  # only bandlimited must be roLTI
                  4: False if cond[4] else True,
              }
              results = {}

              # gather successes/failures
              # [LTI] (no subsampling)
              # shifted input <=> shifted output
              results[0] = (cond[0] == np.allclose(o0b, np.roll(o0a, s)))
              # [LTI]
              # shift of subsampling <=> subsampling of shift
              results[1] = (cond[1] == np.allclose(np.roll(o1a, div(s, d)), o1b))
              # [fLTI]
              # shift of subsampling <=> subsampling of shift; integer `s/d`
              results[2] = (cond[2] == np.allclose(np.roll(o1a, div(s, d)), o1b))
              # [rLTI]
              # upsampled subsampling of shift <=> shift of upsampled subsampling
              results[3] = (cond[3] == np.allclose(np.roll(o1au, s), o1bu))
              # [roLTI] upsampled subsampling of shift <=> shift
              results[4] = (cond[4] == np.allclose(o1bu, o0b))

              # run assertions on non-optionals, append info about the rest
              for i in results:
                  if i != 0:
                      if not optional[i]:
                          assert results[i], _err(i)
                      else:
                          if results[i]:
                              optional_successes[i].append(tuple(cfg.values()))
                  else:
                      assert results[0], (
                          "0:\n" "the operation, unrelated to subsampling, "
                          "isn't LTI")

#%%###########################################################################
# Trivia
# ------
print("Number of optional (unexpected) successes:\n  " +
      "\n  ".join(f"Case {i}: {len(v)}" for i, v in optional_successes.items()))
print("\nExample optional successes for case 2:\n  " +
      ", ".join(list(cfg)) + "\n  " +
      "\n  ".join(str(v) for v in optional_successes[2][:5]))
