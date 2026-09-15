#!/usr/bin/env python3
"""Turn the measured transients into a per-frame render timeline.

v3 - cut on the spikes, not on a grid.  v2 placed every cut on a fitted
113.31 BPM grid.  Measuring the audio at 2.9 ms resolution showed why that
felt wrong: this track grooves in triplets/sixths, and 79 % of its real
transients sit more than 30 ms away from a straight beat (median 153 ms).

So the timing source here is analysis/transients.json - 291 detected attacks,
each refined to its 50 % rise point - and every cut, slam, glitch, whip and
flash is placed on one of them.  No quantisation is applied at any stage; the
only rounding is to the video frame (33 ms), and a cut always lands on the
frame that contains the attack, never after it.

  out: analysis/timeline.csv, analysis/edit_plan.md
"""
import json, sys, math
import numpy as np

FPS = 30

ORDER = ("time mode cx cy cz tx ty tz roll fov shx shy zoom flash expo chroma rblur "
         "disk jet neb star twk bloom grain vign warp tunnel hue sat text talpha "
         "tscale toff fade seed invert contrast disktemp stardens rings "
         "glitch mirror spin shutter lightaz lightel").split()

DEFAULT = dict(time=0, mode=0, cx=0, cy=0, cz=-20, tx=0, ty=0, tz=0, roll=0, fov=48,
               shx=0, shy=0, zoom=1, flash=0, expo=1, chroma=0.0015, rblur=0, disk=1,
               jet=0.25, neb=1.0, star=1.0, twk=0.16, bloom=0.62, grain=0.028, vign=0.45,
               warp=0, tunnel=0, hue=0, sat=1.02, text=-1, talpha=0, tscale=1, toff=0,
               fade=0, seed=17, invert=0, contrast=1.06, disktemp=0.18, stardens=1,
               rings=1, glitch=0, mirror=0, spin=0, shutter=0, lightaz=40, lightel=18)

SHADOW = 2.598

# Subjects the edit can cut to.  A cut that only changes the angle on the same
# object does not read as a cut, so every shot also changes what is on screen.
BH, WARP, TUNNEL, PLANET, STAR, NEB = 0, 1, 2, 3, 4, 5
SUBJ_NAME = {BH: "black hole", WARP: "warp", TUNNEL: "singularity",
             PLANET: "ringed planet", STAR: "star", NEB: "nebula"}

# per subject: (radius range, framing range) - framing is only used by the
# black hole, the others take an explicit fov range
SUBJ_CAM = {
    BH:     dict(r=(6.5, 26.0), frac=(0.22, 0.72)),
    PLANET: dict(r=(3.4, 10.0), fov=(21.0, 40.0)),
    STAR:   dict(r=(2.2, 9.5),  fov=(22.0, 58.0)),
    NEB:    dict(r=(0.5, 16.0), fov=(40.0, 62.0)),
}
ELEV = {'low': 6.0, 'high': 38.0, 'under': -24.0, 'mid': 17.0, 'edge': -6.0}
ELEV_CYCLE = ['low', 'high', 'under', 'edge', 'mid', 'high', 'under', 'low', 'edge', 'mid']


def fov_for(r, frac):
    ang = 2.0 * math.atan2(SHADOW, max(r, 3.2))
    return max(22.0, min(88.0, math.degrees(ang / max(frac, 0.05))))


def spherical(r, el_deg, az_deg):
    el, az = math.radians(el_deg), math.radians(az_deg)
    return (r * math.cos(el) * math.sin(az), r * math.sin(el), -r * math.cos(el) * math.cos(az))


class Shot:
    """One cut.  Most last a single beat, and consecutive shots differ in
    subject, scale and viewpoint - not just in azimuth."""

    def __init__(self, t0, t1, idx, subj, sec, k):
        self.t0, self.t1, self.idx, self.subj, self.sec, self.k = t0, t1, idx, subj, sec, k
        rng = np.random.default_rng(4000 + idx * 7919)
        self.rng = rng
        self.elev = ELEV_CYCLE[idx % len(ELEV_CYCLE)]
        self.el0 = ELEV[self.elev] * float(rng.uniform(0.75, 1.25))
        self.az0 = (idx * 137.5 + float(rng.uniform(-20, 20))) % 360.0
        self.azsweep = float(rng.uniform(6, 16)) * (1 if idx % 3 else -1)
        self.elsweep = float(rng.uniform(-6, 6))
        self.roll0 = float(rng.uniform(-0.12, 0.12))
        self.rollsweep = float(rng.uniform(-0.08, 0.08))
        self.ty = float(rng.uniform(-0.32, 0.32))
        self.push = 0.86 if idx % 2 == 0 else 1.14
        cam = SUBJ_CAM.get(subj)
        if cam:
            lo, hi = cam['r']
            # alternate near and far so the scale really changes at each cut
            u = float(rng.uniform(0.0, 0.45)) if idx % 2 == 0 else float(rng.uniform(0.55, 1.0))
            self.r0 = lo + (hi - lo) * u
            if subj == BH:
                f0, f1 = cam['frac']
                self.frac = f1 - (f1 - f0) * u          # near shot = tight framing
                self.fov = None
            else:
                v0, v1 = cam['fov']
                self.fov = v1 - (v1 - v0) * u
                self.frac = None
        else:
            self.r0, self.frac, self.fov = 20.0, None, 52.0
        # planets need a light; keep it three-quarter to the camera
        self.lightaz = self.az0 + float(rng.uniform(40, 78)) * (1 if idx % 2 else -1)
        self.lightel = float(rng.uniform(10, 32))
        # nebula fly-through: drift the camera through the cloud
        self.nebdir = rng.normal(size=3)
        self.nebdir = self.nebdir / (np.linalg.norm(self.nebdir) + 1e-9)
        self.nebspeed = float(rng.uniform(1.6, 4.2))
        self.label = f"{SUBJ_NAME[subj]} / {self.elev}"

    def cam(self, p):
        r = self.r0 * (1.0 + (self.push - 1.0) * p)
        el = self.el0 + self.elsweep * p
        az = self.az0 + self.azsweep * p
        return r, el, az, self.roll0 + self.rollsweep * p


def main(beatmap_path, out_csv, out_md):
    B = json.load(open(beatmap_path))
    dur = B['duration']
    efps = B['env_fps']
    rms = np.array(B['rms'])

    TR = json.load(open('analysis/transients.json'))
    pk = TR['peaks']
    tt = np.array([p['t'] for p in pk])          # attack times, seconds
    ts = np.array([p['s'] for p in pk])          # strength
    tlow = np.array([p['low'] for p in pk])
    thigh = np.array([p['high'] for p in pk])
    npk = len(pk)

    # strength classes, measured rather than assumed
    q_big = np.percentile(ts, 88)
    q_mid = np.percentile(ts, 55)
    is_low = tlow >= thigh
    q_hi = np.percentile(ts, 68)
    accent = (ts >= q_mid) & is_low              # gets the slam
    hihit = (ts >= q_hi) & ~is_low               # gets the glitch - top third only
    huge = ts >= q_big                           # gets whip + flash

    def near(t, win=0.30):
        """index of the strongest transient within +-win of t"""
        m = np.where(np.abs(tt - t) <= win)[0]
        return int(m[np.argmax(ts[m])]) if len(m) else int(np.argmin(np.abs(tt - t)))

    # --- sections: boundaries verified against the per-bar energy table,
    #     then snapped onto the nearest strong transient ------------------
    RAW = [
        ("intro",      0.00,  10.65, 0.20, dict(pool=[NEB, PLANET, STAR, BH], gap=0.52)),
        ("riser",     10.65,  12.77, 0.55, dict(pool=[BH, WARP, STAR, PLANET], gap=0.34, build=True)),
        ("drop A",    12.77,  23.36, 1.00, dict(pool=[BH, PLANET, BH, STAR, WARP, BH, NEB, PLANET], gap=0.19)),
        ("section B", 23.36,  35.54, 1.00, dict(pool=[BH, STAR, PLANET, WARP, BH, NEB, PLANET, BH], gap=0.19)),
        ("dropout",   35.54,  36.07, 0.00, dict(pool=[BH], gap=9.0, dark=True)),
        ("breakdown", 36.07,  40.31, 0.22, dict(pool=[NEB, PLANET, NEB, BH], gap=0.68, calm=True)),
        ("build",     40.31,  42.42, 0.70, dict(pool=[BH, WARP, STAR, BH], gap=0.28, build=True)),
        ("main",      42.42,  63.60, 1.00, dict(pool=[BH, PLANET, STAR, BH, WARP, NEB, PLANET, BH, STAR, TUNNEL], gap=0.17)),
        ("filter",    63.60,  69.96, 0.60, dict(pool=[TUNNEL, STAR, TUNNEL, NEB, TUNNEL, PLANET], gap=0.30)),
        ("finale",    69.96,  dur,   1.10, dict(pool=[BH, WARP, STAR, BH, TUNNEL, PLANET, BH, WARP], gap=0.16)),
    ]
    SECT = []
    for nme, t0_, t1_, inten, extra in RAW:
        s0 = tt[near(t0_)] if t0_ > 0.05 else 0.0
        SECT.append(dict(n=nme, t0=float(s0), t1=float(t1_), intensity=inten, **extra))
    for i in range(len(SECT) - 1):
        SECT[i]['t1'] = SECT[i + 1]['t0']
    SECT[-1]['t1'] = dur

    # --- pick the cut points straight off the transient list -------------
    cuts = []
    for sec in SECT:
        if sec['intensity'] <= 0.01:
            continue
        last = -9.0
        for i in range(npk):
            t = tt[i]
            if not (sec['t0'] <= t < sec['t1']):
                continue
            g = sec['gap']
            # the biggest hits always get their own cut, even in a dense run
            if t - last >= (g * 0.60 if huge[i] else g) and ts[i] >= (0.09 if huge[i] else 0.13):
                cuts.append(i)
                last = t
    cuts = sorted(set(cuts))

    # --- shots --------------------------------------------------------
    shots = []
    prev = None
    pool_pos = {}
    for n_, ci in enumerate(cuts):
        t0_ = float(tt[ci])
        t1_ = float(tt[cuts[n_ + 1]]) if n_ + 1 < len(cuts) else dur
        sec = next((s for s in SECT if s['t0'] <= t0_ < s['t1']), SECT[-1])
        pool = sec['pool']
        pi = pool_pos.get(sec['n'], 0)
        for _ in range(len(pool)):
            subj = pool[pi % len(pool)]
            pi += 1
            if subj != prev:
                break
        pool_pos[sec['n']] = pi
        prev = subj
        sh = Shot(t0_, t1_, n_, subj, sec, ci)
        sh.peak = ci
        shots.append(sh)

    # ------------------------------------------------------- move scheduling
    # Every move below is hung on a measured transient.  A cut or hit is put on
    # the frame that *contains* the attack, so the picture never changes late.
    nfr = int(math.floor(dur * FPS))
    f_of = lambda t: int(math.floor(t * FPS))
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

    sec_of = lambda t: next((s for s in SECT if s['t0'] <= t < s['t1']), SECT[-1])

    for i in range(npk):
        t = float(tt[i])
        f0 = f_of(t)
        if f0 >= nfr:
            break
        sec = sec_of(t)
        inten = sec['intensity']
        if inten <= 0.01:
            continue
        s = float(ts[i])

        if accent[i] or huge[i]:
            # SLAM - scaled by the measured strength of this attack
            peak = (0.13 + 0.26 * min(s, 1.4)) * inten
            for f, v in env_decay(f0, 9, peak, 2.6):
                zoom[f] = max(zoom[f], 1.0 + v)
                shake[f] = max(shake[f], v * 0.055)
                rblur[f] = max(rblur[f], min(1.0, v * 4.2))
            for f, v in env_decay(f0, 5, 0.030 * inten, 2.0):
                chroma[f] = max(chroma[f], 0.0015 + v)
            moves.append((t, f"slam {s:.2f}"))
        elif hihit[i]:
            # GLITCH on the high-band attacks
            for j in range(2):
                f = f0 + j
                if f < nfr:
                    glitch[f] = max(glitch[f], (0.30 + 0.34 * min(float(thigh[i]), 1.3)) * inten)
                    chroma[f] = max(chroma[f], 0.016 * inten)
            moves.append((t, "glitch"))

        if huge[i] and inten > 0.5:
            for f, v in env_decay(f0, 6, 0.42 * inten, 1.6):
                flash[f] = max(flash[f], v)

    # --- stutter on dense runs: three or more attacks inside 320 ms -------
    i = 0
    while i < npk - 2:
        if tt[i + 2] - tt[i] < 0.32 and sec_of(tt[i])['intensity'] > 0.5:
            f0 = f_of(tt[i])
            for j in range(10):
                f = f0 + j
                if f < nfr:
                    stut[f] = 1 + (j // 2) % 2
            moves.append((float(tt[i]), "stutter"))
            i += 3
        else:
            i += 1

    # --- section changes: whip in, flash, and a speed ramp on the big ones
    BIG = {"drop A", "main", "finale"}
    for sec in SECT[1:]:
        if sec['intensity'] <= 0.01:
            continue
        f0 = f_of(sec['t0'])
        big = sec['n'] in BIG
        for j in range(4):
            f = f0 - 4 + j
            if 0 <= f < nfr:
                w = (j + 1) / 4.0
                spin[f] = 0.42 * w * (1 if hash(sec['n']) % 2 else -1)
                rblur[f] = max(rblur[f], 0.95 * w)
                chroma[f] = max(chroma[f], 0.02 * w)
        for f, v in env_decay(f0, 8, 0.95 if big else 0.55, 1.7):
            flash[f] = max(flash[f], v)
        if big:
            for j in range(10):
                f = f0 - 10 + j
                if 0 <= f < nfr:
                    speed[f] = 0.18 + 0.30 * (j / 10.0)
            for j in range(7):
                f = f0 + j
                if f < nfr:
                    speed[f] = 2.6 - 0.22 * j
            moves.append((sec['t0'], f"ramp + whip -> {sec['n']}"))
        else:
            moves.append((sec['t0'], f"whip -> {sec['n']}"))

    # --- invert frames on the five strongest attacks of the loud sections
    loud = [i for i in range(npk) if sec_of(tt[i])['intensity'] >= 1.0]
    for i in sorted(loud, key=lambda j: -ts[j])[:5]:
        f0 = f_of(tt[i])
        for j in range(2):
            if f0 + j < nfr:
                invert[f0 + j] = 1.0
                chroma[f0 + j] = 0.022
        moves.append((float(tt[i]), "invert"))

    # --- mirrored passages, each starting on an attack -------------------
    for t_start, kind in ((28.0, 1), (55.0, 2), (80.6, 1)):
        i = near(t_start)
        j = near(tt[i] + 2.1)
        mirror[f_of(tt[i]):f_of(tt[j])] = kind
        moves.append((float(tt[i]), f"mirror x{kind}"))

    # --- the beat the track drops out: freeze and go dark ---------------
    dsec = next(s for s in SECT if s['n'] == 'dropout')
    fa, fz = f_of(dsec['t0']), f_of(dsec['t1'])
    speed[fa:fz] = 0.0
    fade[fa:fz] = np.linspace(0.55, 0.88, max(fz - fa, 1))
    moves.append((dsec['t0'], "freeze + blackout"))

    # --- section-wide speed treatment ------------------------------------
    bd = next(s for s in SECT if s['n'] == 'breakdown')
    speed[f_of(bd['t0']):f_of(bd['t1'])] = 0.42
    fl = next(s for s in SECT if s['n'] == 'filter')
    speed[f_of(fl['t0']):f_of(fl['t1'])] = 0.85
    fin = next(s for s in SECT if s['n'] == 'finale')
    speed[f_of(fin['t0']):nfr] = 1.35

    # --- strobe the last two seconds -------------------------------------
    fa = f_of(dur - 2.2)
    for f in range(fa, nfr):
        if (f - fa) % 6 < 2 and f < nfr - 12:
            flash[f] = max(flash[f], 0.55)

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
        Pm['mode'] = sh.subj
        # the black hole keeps one seed so it stays the same object across cuts;
        # planets, stars and nebulae get a fresh one per shot so each cut shows
        # a different body rather than the same one from another angle
        Pm['seed'] = 17 if sh.subj in (BH, WARP, TUNNEL) else (17 + sh.idx * 13) % 997
        dist = 1.0
        if sh.subj in (BH, PLANET, STAR):
            r, el, az, roll = sh.cam(p)
            r /= (1.0 + 0.30 * (zoom[fi] - 1.0))          # the slam shoves the camera in
            cx, cy, cz = spherical(r, el, az)
            Pm.update(cx=cx, cy=cy, cz=cz, tx=0.0, ty=sh.ty, tz=0.0)
            Pm['fov'] = fov_for(r, sh.frac) if sh.subj == BH else sh.fov
            Pm['roll'] = roll
            Pm['lightaz'] = sh.lightaz + 12.0 * p
            Pm['lightel'] = sh.lightel
            if sh.subj == BH:
                dist = min(1.20, max(0.45, (r / 12.0) ** 0.55))
            elif sh.subj == STAR:
                Pm['disk'] = 0.85 * min(1.25, max(0.55, (r / 5.0) ** 0.45))
        elif sh.subj == NEB:
            # fly through the cloud: the camera translates instead of orbiting
            s = sh.r0 + sh.nebspeed * (t - sh.t0)
            base = sh.nebdir * s
            fwd = np.array([math.sin(math.radians(sh.az0)), 0.25 * math.sin(sh.el0 * 0.05),
                            -math.cos(math.radians(sh.az0))])
            fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
            pos = base
            tgt = base + fwd * 6.0
            Pm.update(cx=float(pos[0]), cy=float(pos[1]), cz=float(pos[2]),
                      tx=float(tgt[0]), ty=float(tgt[1]), tz=float(tgt[2]))
            Pm['fov'] = sh.fov
            Pm['roll'] = sh.roll0 + sh.rollsweep * p
        else:                                              # warp / tunnel: screen space
            Pm['fov'] = 52.0
            Pm['roll'] = sh.roll0 + sh.rollsweep * p
        Pm['warp'] = (0.85 + 0.45 * p) if sh.subj == WARP else 0.0
        Pm['tunnel'] = (0.85 + 0.55 * p) if sh.subj == TUNNEL else 0.0

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
            Pm.update(expo=0.74, disk=0.55 * dist, jet=0.06, neb=1.10, sat=0.80,
                      vign=0.58, bloom=0.48, grain=0.038, star=1.05, disktemp=0.30)
            if sh.subj == STAR:
                Pm['disk'] = 0.7
        elif sec['n'].startswith('intro'):
            f = 1.0 - min(1.0, sh.k / 20.0)
            Pm.update(expo=0.80 + 0.12 * (1 - f), disk=(0.45 + 0.60 * (1 - f)) * dist,
                      jet=0.10, neb=1.05, sat=0.94, vign=0.54, bloom=0.50,
                      grain=0.034, star=1.08, disktemp=0.30)
            if sh.subj == STAR:
                Pm['disk'] = 0.8
            elif sh.subj in (PLANET, NEB):
                Pm['expo'] = 0.95
        elif sec.get('dark'):
            Pm.update(expo=0.12, disk=0.5, jet=0.0, neb=0.5, star=0.6, bloom=0.3, vign=0.75)
        else:
            boost = 1.0 + 0.55 * (zoom[fi] - 1.0) / 0.38
            Pm.update(expo=1.02, jet=0.34 * boost, neb=0.95, sat=1.03, vign=0.44,
                      bloom=0.66 * (1 + 0.12 * (boost - 1)), grain=0.026, star=1.0,
                      disktemp=0.18)
            if sh.subj == BH:
                Pm['disk'] = 1.15 * dist * boost
            elif sh.subj == STAR:
                Pm['disk'] = Pm.get('disk', 0.85) * boost
                Pm['bloom'] = 0.52
                Pm['expo'] = 0.96
            elif sh.subj == PLANET:
                Pm['expo'] = 0.90
                Pm['bloom'] = 0.48
                Pm['neb'] = 0.8
            elif sh.subj == NEB:
                Pm['expo'] = 1.0
                Pm['bloom'] = 0.60
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
        elif 78.4 < t < 82.6:
            a = min(1.0, (t - 78.4) / 0.5) * (1.0 - max(0.0, (t - 81.7) / 0.9))
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
    cut_t = np.array([s.t0 for s in shots])
    gaps = np.diff(cut_t)
    dev = [0.0]
    with open(out_md, 'w') as f:
        f.write("# Edit plan v2 - singularity.wav (VUTRA)\n\n")
        f.write(f"- {dur:.2f}s, {nfr} frames @ {FPS} fps, 1080x1920\n")
        f.write(f"- timing source: **{npk} measured transients**, not a tempo grid "
                f"(this track grooves in triplets; 79 % of its attacks sit >30 ms off a "
                f"straight beat)\n")
        f.write(f"- **{len(shots)} shots**, mean shot length **{gaps.mean():.2f}s** "
                f"(shortest {gaps.min():.2f}s), v1 had 37 shots at 2.33s\n")
        f.write(f"- every cut sits on a detected attack; the only rounding is to the "
                f"video frame, and always onto the frame that contains the attack\n")
        f.write(f"- slam on {int((accent|huge).sum())} attacks, glitch on {int(hihit.sum())}, "
                f"{int(huge.sum())} of them carry a flash\n\n")
        f.write("## sections\n\n| time | section | shots | subjects |\n")
        f.write("|------|---------|-------|----------|\n")
        for sec in SECT:
            ss = [s for s in shots if s.sec is sec]
            subs = ", ".join(sorted({SUBJ_NAME[s.subj] for s in ss}))
            f.write(f"| {sec['t0']:.2f}-{sec['t1']:.2f}s | {sec['n']} | {len(ss)} | {subs} |\n")
        from collections import Counter
        cnt = Counter(SUBJ_NAME[s.subj] for s in shots)
        f.write("\n## subject mix\n\n")
        for nme, c in cnt.most_common():
            f.write(f"- {nme}: {c} shots ({c/len(shots)*100:.0f} %)\n")
        rep = sum(1 for i in range(1, len(shots)) if shots[i].subj == shots[i-1].subj)
        f.write(f"\nconsecutive shots on the same subject: {rep}\n")
        f.write("\n## moves\n\n")
        for t, m in sorted(moves):
            f.write(f"- {t:6.2f}s  {m}\n")
    print(f"{nfr} frames, {len(shots)} shots, mean {gaps.mean():.2f}s, min {gaps.min():.2f}s")
    from collections import Counter
    cnt = Counter(SUBJ_NAME[s.subj] for s in shots)
    print("subjects: " + ", ".join(f"{k} {v}" for k, v in cnt.most_common()))
    rep = sum(1 for i in range(1, len(shots)) if shots[i].subj == shots[i-1].subj)
    print(f"same-subject cuts in a row: {rep}")
    print(f"slam {int((accent|huge).sum())}, glitch {int(hihit.sum())}, flash {int(huge.sum())}, "
          f"moves {len(moves)}")
    print(f"cuts: {len(shots)}, all on measured attacks")


main(sys.argv[1], sys.argv[2], sys.argv[3])
