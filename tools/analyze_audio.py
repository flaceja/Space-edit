#!/usr/bin/env python3
"""Beat / structure analysis for the space edit.

Pure numpy+scipy implementation:
  * spectral-flux onset envelope (per frequency band)
  * tempo estimate via autocorrelation + comb scoring of the onset envelope
  * beat tracking via Ellis-style dynamic programming
  * downbeat / bar phase from low-band energy periodicity
  * section boundaries from a self-similarity novelty curve

Writes analysis/beatmap.json used by the renderer.
"""
import json, sys, os
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d, uniform_filter1d

SR_TARGET = 22050
HOP = 256                    # ~11.6 ms frames at 22050
N_FFT = 1024

def load(path):
    sr, x = wavfile.read(path)
    if x.dtype.kind == 'i':
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return sr, x

def stft_mag(x, n_fft=N_FFT, hop=HOP):
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * win
    return np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)

def onset_envelope(S, sr, hop, fmin=0, fmax=None):
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    lo = np.searchsorted(freqs, fmin)
    hi = len(freqs) if fmax is None else np.searchsorted(freqs, fmax)
    B = S[:, lo:hi]
    L = np.log1p(200.0 * B)
    d = np.diff(L, axis=0, prepend=L[:1])
    flux = np.maximum(d, 0).sum(axis=1)
    # subtract local mean -> emphasise transients
    flux = flux - uniform_filter1d(flux, size=int(0.6 * sr / hop))
    flux = np.maximum(flux, 0)
    if flux.max() > 0:
        flux /= flux.max()
    return flux

def tempo_estimate(env, sr, hop, bpm_lo=60, bpm_hi=200):
    """Comb-filter score over candidate BPMs on the autocorrelated envelope."""
    e = env - env.mean()
    n = len(e)
    ac = np.correlate(e, e, mode='full')[n - 1:]
    ac /= (ac[0] + 1e-9)
    fps = sr / hop
    cands = np.arange(bpm_lo, bpm_hi + 0.25, 0.25)
    scores = []
    for bpm in cands:
        period = 60.0 / bpm * fps
        s = 0.0
        for k in (1, 2, 3, 4):           # beat + its multiples => favours real pulse
            lag = period * k
            i = int(round(lag))
            if i >= len(ac) - 1:
                break
            frac = lag - i
            v = ac[i] * (1 - frac) + ac[min(i + 1, len(ac) - 1)] * frac
            s += v / k
        scores.append(s)
    scores = np.array(scores)
    # mild prior towards 90-160 bpm (typical edit range)
    prior = np.exp(-0.5 * ((np.log(cands / 125.0)) / 0.45) ** 2)
    scored = scores * prior
    best = cands[int(np.argmax(scored))]
    return float(best), cands, scored

def beat_track(env, sr, hop, bpm, tightness=100.0):
    """Ellis dynamic-programming beat tracker."""
    fps = sr / hop
    period = 60.0 / bpm * fps
    local = env / (env.max() + 1e-9)
    N = len(local)
    window = np.arange(-int(round(2 * period)), -int(round(period / 2)) + 1)
    window = window[window < 0]
    txwt = -tightness * (np.log(-window / period) ** 2)
    backlink = np.full(N, -1, dtype=int)
    cumscore = local.astype(np.float64).copy()
    for i in range(N):
        cand = window + i
        ok = cand >= 0
        if not ok.any():
            continue
        sc = txwt[ok] + cumscore[cand[ok]]
        j = int(np.argmax(sc))
        if sc[j] > 0:
            cumscore[i] = local[i] + sc[j]
            backlink[i] = cand[ok][j]
    # backtrace from a strong maximum near the end
    tail = cumscore - uniform_filter1d(cumscore, size=int(period * 4))
    lo = int(len(tail) * 0.1)
    start = lo + int(np.argmax(tail[lo:]))
    beats = [start]
    while backlink[beats[-1]] >= 0:
        beats.append(backlink[beats[-1]])
    beats = np.array(beats[::-1])
    # extend the grid forward past the last tracked beat to the end of the track
    if len(beats) > 2:
        ibi = np.median(np.diff(beats))
        nxt = beats[-1] + ibi
        ext = []
        while nxt < N - 1:
            # snap to the strongest onset within +-8% of the period
            w = int(max(1, 0.08 * ibi))
            a, b = int(max(0, nxt - w)), int(min(N, nxt + w + 1))
            k = a + int(np.argmax(local[a:b])) if b > a else int(nxt)
            ext.append(k if local[k] > 0.05 else int(round(nxt)))
            nxt = ext[-1] + ibi if local[ext[-1]] > 0.05 else nxt + ibi
        beats = np.concatenate([beats, np.array(ext, dtype=int)])
    return beats * hop / sr

def band_energy(S, sr, lo, hi):
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    a, b = np.searchsorted(freqs, lo), np.searchsorted(freqs, hi)
    return np.sqrt((S[:, a:b] ** 2).mean(axis=1) + 1e-12)

def novelty_sections(S, sr, hop, min_gap=4.0):
    """Self-similarity novelty (checkerboard kernel) -> structural boundaries."""
    n_mel = 40
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    edges = np.geomspace(40, min(8000, sr / 2 - 1), n_mel + 1)
    idx = [np.searchsorted(freqs, e) for e in edges]
    F = np.stack([np.log1p(S[:, idx[i]:max(idx[i + 1], idx[i] + 1)].mean(axis=1)) for i in range(n_mel)], axis=1)
    # downsample to ~10 fps
    fps = sr / hop
    step = max(1, int(round(fps / 10)))
    F = F[::step]
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    SSM = Fn @ Fn.T
    L = 16
    k = np.zeros((2 * L, 2 * L))
    k[:L, :L] = 1; k[L:, L:] = 1; k[:L, L:] = -1; k[L:, :L] = -1
    g = np.outer(*(2 * [np.exp(-0.5 * (np.linspace(-2, 2, 2 * L)) ** 2)]))
    k *= g
    nov = np.zeros(SSM.shape[0])
    for i in range(L, SSM.shape[0] - L):
        nov[i] = (SSM[i - L:i + L, i - L:i + L] * k).sum()
    nov = np.maximum(nov, 0)
    nov /= (nov.max() + 1e-9)
    t = np.arange(len(nov)) * step * hop / sr
    peaks = []
    thr = 0.25
    mx = maximum_filter1d(nov, size=int(min_gap * 10))
    for i in range(len(nov)):
        if nov[i] == mx[i] and nov[i] > thr:
            if not peaks or t[i] - peaks[-1] >= min_gap:
                peaks.append(float(t[i]))
    return peaks, nov.tolist(), t.tolist()

def main():
    wav = sys.argv[1]
    out = sys.argv[2]
    sr, x = load(wav)
    dur = len(x) / sr
    S = stft_mag(x)
    fps = sr / HOP
    env_full = onset_envelope(S, sr, HOP)
    env_low  = onset_envelope(S, sr, HOP, 20, 250)
    env_hi   = onset_envelope(S, sr, HOP, 3000, 11000)

    bpm, cands, scores = tempo_estimate(env_full, sr, HOP)
    beats = beat_track(env_full, sr, HOP, bpm)
    # refine bpm from the actual beat spacing (median IBI)
    ibi = np.diff(beats)
    bpm_ref = 60.0 / float(np.median(ibi)) if len(ibi) else bpm

    # per-frame RMS on the same grid as the STFT frames
    nfr = S.shape[0]
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(nfr)[:, None]
    rms_frames = np.sqrt(np.maximum((x[idx] ** 2).mean(axis=1), 0.0))
    rms = rms_frames
    t_frames = np.arange(nfr) * HOP / sr

    bass = band_energy(S, sr, 20, 160)
    mid  = band_energy(S, sr, 160, 2000)
    high = band_energy(S, sr, 2000, 11000)

    # ---- lock a constant-tempo grid (this track drifts < 15 ms over 86 s) ----
    kk = np.arange(len(beats))
    (period, tstart), *_ = np.linalg.lstsq(np.vstack([kk, np.ones_like(kk)]).T, beats, rcond=None)
    resid = beats - (tstart + period*kk)
    grid = tstart + period*np.arange(int((dur - tstart)/period) + 1)

    # ---- downbeat phase ------------------------------------------------
    # A kick-energy vote is ambiguous here: in a half-time feel phases 0
    # and 2 both carry kicks.  Instead find the beats where the arrangement
    # actually changes (largest two-bar energy step) and let those vote -
    # section changes land on bar lines, so their phase is the bar phase.
    def energy_step(k, bars=2):
        w = int(bars*4*period*fps)
        i = int(grid[k]*fps)
        if i-w < 0 or i+w > len(rms_frames):
            return 0.0
        return abs(rms_frames[i:i+w].mean() - rms_frames[i-w:i].mean())
    steps = np.array([energy_step(k) for k in range(len(grid))])
    order = np.argsort(steps)[::-1]
    picks, votes = [], np.zeros(4)
    for k in order:
        if steps[k] <= 0:
            break
        if any(abs(k-q) < 6 for q in picks):     # one vote per transition
            continue
        picks.append(int(k))
        votes[k % 4] += steps[k]
        if len(picks) >= 6:
            break
    phase = int(np.argmax(votes))
    downbeats = grid[phase::4]
    bpm_grid = 60.0/period

    # energy per beat (drives the renderer)
    beats = grid            # everything downstream uses the locked grid
    bi = np.clip((beats * fps).astype(int), 0, len(env_low) - 1)
    strength = env_low[bi] + 0.5 * env_full[bi]

    def at(sig, times):
        i = np.clip((times * fps).astype(int), 0, len(sig) - 1)
        return sig[i]
    nb = lambda v: (v / (np.percentile(v, 99) + 1e-9)).clip(0, 1)

    sections, nov, nov_t = novelty_sections(S, sr, HOP)

    data = dict(
        file=os.path.basename(wav), sr=sr, duration=dur,
        bpm=round(float(bpm_grid), 4), bpm_tracked=round(float(bpm_ref), 3),
        beat_period=round(float(period), 6), grid_t0=round(float(tstart), 6),
        grid_residual_ms=round(float(np.abs(resid).max()*1000), 2),
        n_beats=len(beats), beats=[round(float(b), 4) for b in beats],
        downbeat_phase=phase,
        transition_beats=[int(k) for k in sorted(picks)],
        downbeats=[round(float(b), 4) for b in downbeats],
        beat_strength=[round(float(v), 4) for v in nb(strength)],
        beat_bass=[round(float(v), 4) for v in nb(at(bass, beats))],
        beat_high=[round(float(v), 4) for v in nb(at(high, beats))],
        sections=[round(s, 3) for s in sections],
        env_fps=fps,
        env=[round(float(v), 4) for v in env_full],
        env_low=[round(float(v), 4) for v in env_low],
        env_high=[round(float(v), 4) for v in env_hi],
        bass=[round(float(v), 5) for v in nb(bass)],
        mid=[round(float(v), 5) for v in nb(mid)],
        high=[round(float(v), 5) for v in nb(high)],
        rms_fps=fps,
        rms=[round(float(v), 5) for v in nb(rms)],
    )
    with open(out, 'w') as f:
        json.dump(data, f)
    # console report
    print(f"duration {dur:.2f}s  bpm {bpm_grid:.3f} (tracked {bpm_ref:.2f})  beats {len(grid)}  bars {len(downbeats)}")
    print(f"grid t0 {tstart:.4f}s  period {period*1000:.2f} ms  max residual {np.abs(resid).max()*1000:.1f} ms")
    print(f"downbeat phase {phase} -> bar lines at beat k = {phase} mod 4")
    print("  transition beats: " + ", ".join(f"k={k} ({grid[k]:.2f}s, step {steps[k]:.3f})" for k in sorted(picks)))
    print("  phase votes: " + ", ".join(f"{p}:{votes[p]:.3f}" for p in range(4)))
    print("sections:", ", ".join(f"{s:.2f}" for s in sections))
    # coarse energy map, one line per 4 bars
    bar_t = downbeats
    print("\nbar  time     k   bass  mid   high  rms")
    for i, t in enumerate(bar_t):
        j = min(int(t * fps), len(bass) - 1)
        w = slice(j, min(j + int(fps * period * 4), len(bass)))
        print(f"{i:3d} {t:6.2f} {i*4+phase:4d}  {nb(bass)[w].mean():.2f}  {nb(mid)[w].mean():.2f}  "
              f"{nb(high)[w].mean():.2f}  {nb(rms)[w].mean():.2f}  {'#'*int(nb(rms)[w].mean()*34)}")

main()
