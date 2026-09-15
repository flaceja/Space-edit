#!/usr/bin/env python3
"""High resolution transient detection - find the actual spikes in the audio.

The edit is cut on these times directly, not on a quantised tempo grid.
Resolution 2.9 ms (hop 128 at 44.1 kHz); each peak is refined to the point
where the attack crosses half of its rise, which is what the ear registers
as "the hit", and is a few ms earlier than the flux maximum.
"""
import sys, json
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d, uniform_filter1d

HOP, NFFT = 128, 512


def main(wav, out):
    sr, x = wavfile.read(wav)
    x = x.astype(np.float32) / (np.iinfo(np.int16).max if x.dtype.kind == 'i' else 1.0)
    if x.ndim > 1:
        x = x.mean(axis=1)
    dur = len(x) / sr
    win = np.hanning(NFFT).astype(np.float32)
    nfr = 1 + (len(x) - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(nfr)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * win, axis=1)).astype(np.float32)
    freqs = np.fft.rfftfreq(NFFT, 1.0 / sr)
    fps = sr / HOP

    def flux(lo, hi):
        a, b = np.searchsorted(freqs, lo), np.searchsorted(freqs, hi)
        L = np.log1p(400.0 * S[:, a:b])
        d = np.maximum(np.diff(L, axis=0, prepend=L[:1]), 0).sum(axis=1)
        d = d - uniform_filter1d(d, size=int(0.35 * fps))
        return np.maximum(d, 0)

    f_low = flux(20, 180)        # kick / sub
    f_mid = flux(180, 2000)      # body / snare
    f_high = flux(2000, 16000)   # hats / transient sparkle
    f_all = f_low * 1.6 + f_mid + f_high * 0.9
    n = lambda v: v / (np.percentile(v, 99.7) + 1e-9)
    f_all, f_low, f_mid, f_high = n(f_all), n(f_low), n(f_mid), n(f_high)

    # --- peak picking ---------------------------------------------------
    min_sep = int(0.075 * fps)
    mx = maximum_filter1d(f_all, size=2 * min_sep + 1)
    floor = uniform_filter1d(f_all, size=int(1.5 * fps)) * 1.4 + 0.045
    cand = np.where((f_all == mx) & (f_all > floor))[0]

    peaks = []
    for i in cand:
        pk = f_all[i]
        # walk back to where the attack starts, then take the 50 % rise point
        j = i
        while j > 0 and f_all[j - 1] < f_all[j] and i - j < int(0.05 * fps):
            j -= 1
        half = pk * 0.5
        t_i = i
        for q in range(j, i + 1):
            if f_all[q] >= half:
                if q > j and f_all[q] > f_all[q - 1]:
                    frac = (half - f_all[q - 1]) / (f_all[q] - f_all[q - 1] + 1e-9)
                    t_i = (q - 1) + frac
                else:
                    t_i = q
                break
        t = float(t_i) / fps
        peaks.append(dict(t=round(t, 4), s=round(float(pk), 4),
                          low=round(float(f_low[i]), 4),
                          mid=round(float(f_mid[i]), 4),
                          high=round(float(f_high[i]), 4)))
    peaks.sort(key=lambda p: p['t'])
    json.dump(dict(duration=dur, fps=fps, n=len(peaks), peaks=peaks), open(out, 'w'))

    print(f"{len(peaks)} transients in {dur:.2f}s = {len(peaks)/dur:.2f}/s")
    st = np.array([p['s'] for p in peaks])
    print(f"strength: median {np.median(st):.3f}, 75th {np.percentile(st,75):.3f}, max {st.max():.3f}")
    ib = np.diff([p['t'] for p in peaks])
    print(f"gap between transients: median {np.median(ib)*1000:.0f} ms, min {ib.min()*1000:.0f} ms")
    # how well does a constant 113.31 BPM grid explain them?
    P, t0 = 60/113.307, 0.0586
    off = np.array([(p['t']-t0)/P for p in peaks])
    r1 = np.abs(off - np.round(off))*P*1000
    r2 = np.abs(off*2 - np.round(off*2))*P*500
    r3 = np.abs(off*3 - np.round(off*3))*P/3*1000
    print(f"vs grid  : within a beat {np.median(r1):.0f} ms median, half beat {np.median(r2):.0f} ms, "
          f"triplet {np.median(r3):.0f} ms")
    print(f"share farther than 30 ms from a grid beat: {(r1>30).mean()*100:.0f} %  "
          f"| from a half beat: {(r2>30).mean()*100:.0f} %")


main(sys.argv[1], sys.argv[2])
