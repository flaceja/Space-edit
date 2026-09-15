#!/usr/bin/env python3
"""Turn analysis/beatmap.json into a per-frame render timeline.

v2 - the edit brain.  v1 cut once per bar and answered every one of the 163
beats with the same zoom pulse, which reads as a reaction to music rather than
an edit.  This version:

  * cuts on the beat (half beat in the peaks), ~130 shots instead of 37
  * reserves the big gesture for beats whose measured low-band onset is
    actually strong - the rest get a cut and nothing else
  * has a vocabulary instead of one move: slam, whip, glitch, stutter, strobe,
    invert flash, mirror, freeze, speed ramp, scene stab
  * drives the scene clock through a speed curve, so ramps and freezes hit the
    disk rotation, the star streaks and the tunnel scroll, not just the camera

  out: analysis/timeline.csv   (one row per frame)
       analysis/edit_plan.md   (shot list + move list)
"""
import json, sys, math
import numpy as np

FPS = 30

ORDER = ("time mode cx cy cz tx ty tz roll fov shx shy zoom flash expo chroma rblur "
         "disk jet neb star twk bloom grain vign warp tunnel hue sat text talpha "
         "tscale toff fade seed invert contrast disktemp stardens rings "
         "glitch mirror spin shutter").split()

DEFAULT = dict(time=0, mode=0, cx=0, cy=0, cz=-20, tx=0, ty=0, tz=0, roll=0, fov=48,
               shx=0, shy=0, zoom=1, flash=0, expo=1, chroma=0.0015, rblur=0, disk=1,
               jet=0.25, neb=1.0, star=1.0, twk=0.16, bloom=0.62, grain=0.028, vign=0.45,
               warp=0, tunnel=0, hue=0, sat=1.02, text=-1, talpha=0, tscale=1, toff=0,
               fade=0, seed=17, invert=0, contrast=1.06, disktemp=0.18, stardens=1,
               rings=1, glitch=0, mirror=0, spin=0, shutter=0)

SHADOW = 2.598

# camera archetypes: (radius, framing, elevation).  Consecutive shots step
# through this cycle so every cut changes scale AND viewpoint, never just angle.
SIZE = {'tight': (8.6, 0.70), 'mid': (13.0, 0.46), 'wide': (22.0, 0.26), 'far': (34.0, 0.17)}
ELEV = {'low': 6.0, 'high': 38.0, 'under': -24.0, 'mid': 17.0, 'edge': -6.0}
CYCLE = [('tight', 'low'), ('wide', 'high'), ('mid', 'under'), ('tight', 'edge'),
         ('wide', 'mid'), ('mid', 'high'), ('tight', 'under'), ('wide', 'low'),
         ('mid', 'edge'), ('tight', 'high')]


def fov_for(r, frac):
    ang = 2.0 * math.atan2(SHADOW, max(r, 3.2))
    return max(22.0, min(88.0, math.degrees(ang / max(frac, 0.05))))


def spherical(r, el_deg, az_deg):
    el, az = math.radians(el_deg), math.radians(az_deg)
    return (r * math.cos(el) * math.sin(az), r * math.sin(el), -r * math.cos(el) * math.cos(az))


class Shot:
    """One camera setup, one cut.  Short by design - most last a single beat."""
    def __init__(self, t0, t1, idx, mode=0, size=None, elev=None, push=None, **kw):
        self.t0, self.t1, self.idx, self.mode, self.kw = t0, t1, idx, mode, kw
        rng = np.random.default_rng(1000 + idx * 7919)
        size = size or CYCLE[idx % len(CYCLE)][0]
        elev = elev or CYCLE[idx % len(CYCLE)][1]
        self.label = f"{size}/{elev}"
        r0, frac = SIZE[size]
        self.r0 = r0 * float(rng.uniform(0.88, 1.14))
        self.frac = frac * float(rng.uniform(0.92, 1.10))
        self.el0 = ELEV[elev] * float(rng.uniform(0.80, 1.20))
        self.az0 = (idx * 137.5 + float(rng.uniform(-18, 18))) % 360.0
        # intrinsic motion: alternate hard push-in and pull-out, always moving
        self.push = push if push is not None else (0.86 if idx % 2 == 0 else 1.13)
        self.azsweep = float(rng.uniform(7, 17)) * (1 if idx % 3 else -1)
        self.elsweep = float(rng.uniform(-5, 5))
        self.roll0 = float(rng.uniform(-0.11, 0.11))
        self.rollsweep = float(rng.uniform(-0.07, 0.07))
        self.ty = float(rng.uniform(-0.34, 0.34))

    def cam(self, p):
        r = self.r0 * (1.0 + (self.push - 1.0) * p)
        el = self.el0 + self.elsweep * p
        az = self.az0 + self.azsweep * p
        return r, el, az, self.roll0 + self.rollsweep * p


def main(beatmap_path, out_csv, out_md):
    B = json.load(open(beatmap_path))
    dur = B['duration']
    P = B['beat_period']
    gt0 = B['grid_t0']
    beats = np.array(B['beats'])
    nbeats = len(beats)
    bstr = np.array(B['beat_strength'])
    bbass = np.array(B['beat_bass'])
    bhigh = np.array(B['beat_high'])
    efps = B['env_fps']
    env = np.array(B['env'])
    rms = np.array(B['rms'])

    beat_t = lambda k: gt0 + P * k          # fractional k allowed (half beats)
    bar = lambda n: beat_t(4 * n)

    # --- accent classification --------------------------------------------
    # A beat only earns the big gesture if its measured low-band onset is in
    # the top of the distribution.  Everything else gets a cut, nothing more.
    acc_thr = np.percentile(bstr, 62)
    hi_thr = np.percentile(bhigh, 66)
    accent = bstr >= acc_thr
    hihit = (bhigh >= hi_thr) & ~accent
    # fills: beats packed with onsets -> stutter/roll material
    dens = np.array([float((env[int(beat_t(k) * efps):int(beat_t(k + 1) * efps)] > 0.15).sum())
                     for k in range(nbeats)])
    fill = dens >= np.percentile(dens, 92)

    # --- section map (bar numbers verified against the per-bar energy table)
    SECT = [
        # k0,  k1,  cut(beats), mode, look
        dict(n="intro",     k0=0,   k1=12,  cut=4.0,  intensity=0.20),
        dict(n="intro b",   k0=12,  k1=20,  cut=2.0,  intensity=0.28),
        dict(n="riser",     k0=20,  k1=24,  cut=1.0,  intensity=0.55, build=True),
        dict(n="drop A",    k0=24,  k1=44,  cut=1.0,  intensity=1.00),
        dict(n="section B", k0=44,  k1=67,  cut=1.0,  intensity=1.00),
        dict(n="dropout",   k0=67,  k1=68,  cut=1.0,  intensity=0.0,  dark=True),
        dict(n="breakdown", k0=68,  k1=76,  cut=2.0,  intensity=0.22, calm=True),
        dict(n="build",     k0=76,  k1=80,  cut=1.0,  intensity=0.70, build=True),
        dict(n="main",      k0=80,  k1=120, cut=1.0,  intensity=1.00),
        dict(n="filter",    k0=120, k1=132, cut=1.0,  intensity=0.60, inside=True),
        dict(n="finale",    k0=132, k1=164, cut=1.0,  intensity=1.10),
    ]
    # bars that cut on the half beat (peaks and the run-out)
    HALF = set()
    for a, b in [(96, 100), (108, 112), (116, 120), (148, 152), (156, 164)]:
        HALF.update(range(a, b))
    # single-beat scene stabs: warp (1) or tunnel (2) inserted between shots
    STAB = {31: 1, 39: 1, 51: 1, 59: 1, 63: 2, 86: 1, 95: 1, 103: 1, 111: 2,
            119: 1, 137: 1, 143: 1, 151: 2, 159: 1}
    # whole-bar treatments
    MIRROR_BARS = {13: 1, 26: 2, 38: 1}
    # sections that live inside the singularity
    def mode_for(k, sec):
        if k in STAB:
            return STAB[k]
        if sec['n'] == 'filter':
            return 2 if (k // 2) % 3 != 2 else 0      # flash back outside now and then
        if sec['n'] == 'finale' and 152 <= k < 160:
            return 2
        if sec['n'] == 'finale' and 148 <= k < 152:
            return 1
        return 0

    # ---------------------------------------------------------------- shots
    shots = []
    idx = 0
    for sec in SECT:
        k = sec['k0']
        while k < sec['k1']:
            step = 0.5 if int(k) in HALF else sec['cut']
            k2 = min(k + step, sec['k1'])
            m = mode_for(int(k), sec)
            shots.append(Shot(beat_t(k), beat_t(k2), idx, mode=m, sec=sec, k=int(k)))
            idx += 1
            k = k2
    for s in shots:                       # shots know their section / start beat
        s.sec = s.kw['sec']; s.k = s.kw['k']

    # ------------------------------------------------------- move scheduling
    nfr = int(math.floor(dur * FPS))
    f_of = lambda t: int(round(t * FPS))
    zoom = np.ones(nfr); shake = np.zeros(nfr); rblur = np.zeros(nfr)
    spin = np.zeros(nfr); glitch = np.zeros(nfr); chroma = np.full(nfr, 0.0015)
    flash = np.zeros(nfr); invert = np.zeros(nfr); mirror = np.zeros(nfr)
    fade = np.zeros(nfr); speed = np.ones(nfr); stut = np.zeros(nfr, dtype=int)
    moves = []

    def env_decay(f0, n, peak, tau_frames):
        for i in range(n):
            f = f0 + i
            if 0 <= f < nfr:
                yield f, peak * math.exp(-i / tau_frames)

    for k in range(nbeats):
        t = beat_t(k)
        f0 = f_of(t)
        if f0 >= nfr:
            break
        sec = next((s for s in SECT if s['k0'] <= k < s['k1']), SECT[-1])
        inten = sec['intensity']
        if inten <= 0.01:
            continue
        s = float(bstr[k])

        if accent[k]:
            # SLAM: one big gesture, not a constant pulse
            peak = (0.22 + 0.16 * s) * inten
            for f, v in env_decay(f0, 9, peak, 2.6):
                zoom[f] = max(zoom[f], 1.0 + v)
                shake[f] = max(shake[f], v * 0.055)
                rblur[f] = max(rblur[f], min(1.0, v * 4.2))
            for f, v in env_decay(f0, 5, 0.030 * inten, 2.0):
                chroma[f] = max(chroma[f], 0.0015 + v)
            moves.append((t, "slam"))
        elif hihit[k]:
            # GLITCH: 2 frames of slice tear on the hats/snare accents
            for i in range(2):
                f = f0 + i
                if f < nfr:
                    glitch[f] = max(glitch[f], (0.34 + 0.30 * float(bhigh[k])) * inten)
                    chroma[f] = max(chroma[f], 0.016 * inten)
            moves.append((t, "glitch"))

        if fill[k] and inten > 0.5:
            # STUTTER: the shot alternates with its neighbour every 2 frames
            for i in range(10):
                f = f0 + i
                if f < nfr:
                    stut[f] = 1 + (i // 2) % 2
            moves.append((t, "stutter"))

    # whip into every section change, plus a speed ramp around the big ones
    SECT_K = [24, 44, 68, 76, 80, 120, 132]
    BIG = {24, 80, 132}
    for k in SECT_K:
        f0 = f_of(beat_t(k))
        for i in range(4):                       # whip out of the old shot
            f = f0 - 4 + i
            if 0 <= f < nfr:
                w = (i + 1) / 4.0
                spin[f] = 0.42 * w * (1 if k % 8 else -1)
                rblur[f] = max(rblur[f], 0.95 * w)
                chroma[f] = max(chroma[f], 0.02 * w)
        flash[f0:f0 + 1] = 0.95 if k in BIG else 0.55
        for f, v in env_decay(f0, 8, 0.95 if k in BIG else 0.55, 1.7):
            flash[f] = max(flash[f], v)
        if k in BIG:                             # ramp down, then snap
            for i in range(10):
                f = f0 - 10 + i
                if 0 <= f < nfr:
                    speed[f] = 0.18 + 0.30 * (i / 10.0)
            for i in range(7):
                f = f0 + i
                if f < nfr:
                    speed[f] = 2.6 - 0.22 * i
            moves.append((beat_t(k), "ramp+whip"))
        else:
            moves.append((beat_t(k), "whip"))

    # invert flashes on selected hits
    for k in (56, 104, 144, 152, 160):
        f0 = f_of(beat_t(k))
        for i in range(2):
            if f0 + i < nfr:
                invert[f0 + i] = 1.0
                chroma[f0 + i] = 0.022
        moves.append((beat_t(k), "invert"))

    # mirrored bars
    for b, kind in MIRROR_BARS.items():
        a, z = f_of(bar(b)), f_of(bar(b + 1))
        mirror[a:z] = kind
        moves.append((bar(b), f"mirror x{kind}"))

    # the one beat the track drops out: freeze the scene and kill the light
    fa, fz = f_of(beat_t(67)), f_of(beat_t(68))
    speed[fa:fz] = 0.0
    fade[fa:fz] = np.linspace(0.55, 0.88, fz - fa)
    moves.append((beat_t(67), "freeze + blackout"))

    # strobe out the last bar
    fa = f_of(bar(40))
    for f in range(fa, nfr):
        if (f - fa) % 6 < 2 and f < nfr - 12:
            flash[f] = max(flash[f], 0.55)

    # breakdown runs in slow motion
    speed[f_of(beat_t(68)):f_of(beat_t(76))] = 0.42
    # the filter section drifts, the finale over-cranks
    speed[f_of(beat_t(120)):f_of(beat_t(132))] = 0.85
    speed[f_of(beat_t(132)):nfr] = 1.35

    # ---------------------------------------------------------- per frame
    scene_t = 0.0
    rows = []
    shot_i = 0
    for fi in range(nfr):
        t = fi / FPS
        scene_t += speed[fi] / FPS
        while shot_i + 1 < len(shots) and t >= shots[shot_i + 1].t0:
            shot_i += 1
        sh = shots[shot_i]
        if stut[fi] == 2 and shot_i + 1 < len(shots):
            sh = shots[shot_i + 1]          # stutter: flip between two shots
        sec = sh.sec
        inten = sec['intensity']
        p = (t - sh.t0) / max(sh.t1 - sh.t0, 1e-6)
        p = min(max(p, 0.0), 1.0)

        i_env = min(int(t * efps), len(rms) - 1)
        E = float(rms[i_env])

        Pm = dict(DEFAULT)
        Pm['time'] = scene_t
        Pm['mode'] = sh.mode
        if sh.mode == 0:
            r, el, az, roll = sh.cam(p)
            r /= (1.0 + 0.30 * (zoom[fi] - 1.0))
            cx, cy, cz = spherical(r, el, az)
            Pm.update(cx=cx, cy=cy, cz=cz, tx=0.0, ty=sh.ty, tz=0.0)
            Pm['fov'] = fov_for(r, sh.frac)
            Pm['roll'] = roll
            dist = min(1.20, max(0.45, (r / 12.0) ** 0.55))
        else:
            Pm['fov'] = 52.0
            Pm['roll'] = sh.roll0 + sh.rollsweep * p
            dist = 1.0
        Pm['warp'] = (0.85 + 0.45 * p) if sh.mode == 1 else 0.0
        Pm['tunnel'] = (0.85 + 0.55 * p) if sh.mode == 2 else 0.0

        amp = shake[fi] * (0.6 + 0.8 * E)
        Pm['shx'] = amp * (0.72 * math.sin(t * 63.0) + 0.28 * math.sin(t * 101.0 + 1.3))
        Pm['shy'] = amp * (0.72 * math.sin(t * 71.0 + 0.7) + 0.28 * math.sin(t * 117.0))
        Pm['roll'] += 0.05 * shake[fi] * math.sin(t * 37.0)
        Pm['zoom'] = zoom[fi]
        Pm['rblur'] = rblur[fi]
        Pm['spin'] = spin[fi]
        Pm['glitch'] = glitch[fi]
        Pm['chroma'] = chroma[fi]
        Pm['flash'] = flash[fi]
        Pm['invert'] = invert[fi]
        Pm['mirror'] = mirror[fi]

        # ---- look per section
        if sec.get('calm'):
            Pm.update(expo=0.68, disk=0.55 * dist, jet=0.06, neb=1.15, sat=0.74,
                      vign=0.60, bloom=0.46, grain=0.040, star=1.05, disktemp=0.30)
        elif sec['n'].startswith('intro'):
            f = 1.0 - min(1.0, sh.k / 20.0)
            Pm.update(expo=0.78 + 0.10 * (1 - f), disk=(0.40 + 0.55 * (1 - f)) * dist,
                      jet=0.10, neb=1.10, sat=0.94, vign=0.56, bloom=0.50,
                      grain=0.036, star=1.08, disktemp=0.30)
        elif sec.get('dark'):
            Pm.update(expo=0.12, disk=0.5, jet=0.0, neb=0.5, star=0.6, bloom=0.3, vign=0.75)
        else:
            boost = 1.0 + 0.55 * (zoom[fi] - 1.0) / 0.38
            Pm.update(expo=1.02, disk=1.15 * dist * boost, jet=0.34 * boost,
                      neb=0.95, sat=1.03, vign=0.44, bloom=0.66 * (1 + 0.12 * (boost - 1)),
                      grain=0.026, star=1.0, disktemp=0.18)
        if sec.get('build'):
            Pm['chroma'] += 0.009 * p * p
            Pm['rblur'] = max(Pm['rblur'], 0.5 * p * p)
            Pm['bloom'] *= 1.0 + 0.3 * p
        if sec.get('inside'):
            Pm['bloom'] = 0.68
            Pm['grain'] = 0.045
        Pm['contrast'] = 1.06 + 0.06 * (zoom[fi] - 1.0) / 0.38
        Pm['twk'] = 0.16

        # ---- title cards
        if 2.0 < t < 11.6:
            a = min(1.0, (t - 2.0) / 1.2) * (1.0 - max(0.0, (t - 9.8) / 1.5))
            Pm['text'] = 0
            Pm['talpha'] = max(0.0, a) * 0.97
            Pm['tscale'] = 1.0 + 0.05 * (zoom[fi] - 1.0)
            Pm['toff'] = -0.13
        elif bar(37) < t < bar(39):
            a = min(1.0, (t - bar(37)) / 0.5) * (1.0 - max(0.0, (t - (bar(39) - 0.9)) / 0.9))
            Pm['text'] = 1
            Pm['talpha'] = max(0.0, a) * 0.95
            Pm['tscale'] = 0.85
            Pm['toff'] = 0.30

        fd = fade[fi]
        if t < 0.55:
            fd = max(fd, 1.0 - t / 0.55)
        if t > dur - 1.4:
            fd = max(fd, min(1.0, (t - (dur - 1.4)) / 1.35))
        Pm['fade'] = fd

        rows.append(",".join(f"{float(Pm[k]):.5f}" for k in ORDER))

    open(out_csv, 'w').write("\n".join(rows) + "\n")

    # ------------------------------------------------------------- report
    cuts = [s.t0 for s in shots]
    gaps = np.diff(cuts)
    dev = [min(abs(c - (gt0 + P * round((c - gt0) / P * 2) / 2)) for c in cuts[1:])]
    with open(out_md, 'w') as f:
        f.write("# Edit plan v2 - singularity.wav (VUTRA)\n\n")
        f.write(f"- {dur:.2f}s, {nfr} frames @ {FPS} fps, 1080x1920, {B['bpm']:.2f} BPM\n")
        f.write(f"- **{len(shots)} shots**, mean shot length **{gaps.mean():.2f}s** "
                f"(shortest {gaps.min():.2f}s), v1 had 37 shots at 2.33s\n")
        f.write(f"- cuts land on beats/half beats, max deviation {max(dev)*1000:.1f} ms\n")
        f.write(f"- slams on {int(accent.sum())}/{nbeats} beats (top 38% of measured "
                f"low-band onset), glitch on {int(hihit.sum())}, stutter on {int(fill.sum())} fills\n\n")
        f.write("## sections\n\n| bars | beats | section | cut every | shots |\n")
        f.write("|------|-------|---------|-----------|-------|\n")
        for sec in SECT:
            ns = sum(1 for s in shots if s.sec is sec)
            f.write(f"| {sec['k0']//4}-{sec['k1']//4} | {sec['k0']}-{sec['k1']} | {sec['n']} | "
                    f"{sec['cut']:g} beat | {ns} |\n")
        f.write("\n## moves\n\n")
        for t, m in sorted(moves):
            f.write(f"- {t:6.2f}s  {m}\n")
    print(f"{nfr} frames, {len(shots)} shots, mean {gaps.mean():.2f}s, min {gaps.min():.2f}s")
    print(f"slam {int(accent.sum())}, glitch {int(hihit.sum())}, stutter-fills {int(fill.sum())}, "
          f"moves {len(moves)}")
    print(f"max cut deviation from a (half) beat: {max(dev)*1000:.1f} ms")


main(sys.argv[1], sys.argv[2], sys.argv[3])
