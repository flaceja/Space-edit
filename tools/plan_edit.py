#!/usr/bin/env python3
"""Turn analysis/beatmap.json into a per-frame render timeline.

Every cut sits on a downbeat, every zoom punch / flare / shake impulse is
driven by the measured onset strength of the beat it belongs to, and the
shot list follows the sections found by the structure analysis.

  out: analysis/timeline.csv   (one row per frame, NPARAM floats)
       analysis/edit_plan.md   (human readable shot list)
"""
import json, sys, math
import numpy as np

FPS = 30

ORDER = ("time mode cx cy cz tx ty tz roll fov shx shy zoom flash expo chroma rblur "
         "disk jet neb star twk bloom grain vign warp tunnel hue sat text talpha "
         "tscale toff fade seed invert contrast disktemp stardens rings").split()

DEFAULT = dict(time=0, mode=0, cx=0, cy=0, cz=-20, tx=0, ty=0, tz=0, roll=0, fov=48,
               shx=0, shy=0, zoom=1, flash=0, expo=1, chroma=0.0015, rblur=0, disk=1,
               jet=0.25, neb=1.0, star=1.0, twk=0.35, bloom=0.62, grain=0.028, vign=0.45,
               warp=0, tunnel=0, hue=0, sat=1.02, text=-1, talpha=0, tscale=1, toff=0,
               fade=0, seed=17, invert=0, contrast=1.06, disktemp=0.18, stardens=1, rings=1)


class Shot:
    """One camera setup. Values may be a scalar or a (start, end) pair."""
    def __init__(self, t0, t1, label="", **kw):
        self.t0, self.t1, self.label, self.kw = t0, t1, label, kw

    def val(self, key, p, default=None):
        v = self.kw.get(key, default)
        if v is None:
            return None
        if isinstance(v, (tuple, list)):
            return v[0] + (v[1]-v[0])*p
        return v


SHADOW = 2.598          # apparent radius of the shadow in units of rs


def fov_for(r, frac):
    """Vertical fov (deg) that makes the shadow cover `frac` of frame height."""
    ang = 2.0*math.atan2(SHADOW, max(r, 3.2))      # angular diameter of the shadow
    return max(22.0, min(88.0, math.degrees(ang/max(frac, 0.05))))


def spherical(r, el_deg, az_deg):
    el, az = math.radians(el_deg), math.radians(az_deg)
    return (r*math.cos(el)*math.sin(az), r*math.sin(el), -r*math.cos(el)*math.cos(az))


def main(beatmap_path, out_csv, out_md):
    B = json.load(open(beatmap_path))
    dur = B['duration']
    beats = np.array(B['beats'])
    bstr = np.array(B['beat_strength'])
    bbass = np.array(B['beat_bass'])
    efps = B['env_fps']
    env_hi = np.array(B['env_high'])
    rms = np.array(B['rms'])

    # ---- locked grid straight from the analyser --------------------------
    period = B['beat_period']
    gt0 = B['grid_t0']
    phase = B['downbeat_phase']
    gbeats = np.array(B['beats'])            # already the locked grid
    downs = np.array(B['downbeats'])
    bpm = B['bpm']
    grid_ok = B['grid_residual_ms'] < 45.0
    resid_ms = B['grid_residual_ms']

    def bar(i):
        return float(downs[min(i, len(downs)-1)])

    # per-beat punch strength from the measured onsets
    ps = np.clip(0.34 + 0.66*np.maximum(bbass, bstr*0.8), 0.18, 1.0)

    # ---------------------------------------------------------------- shots
    S, END = [], dur
    add = lambda *a, **k: S.append(Shot(*a, **k))
    beat = lambda k: float(gt0 + period*k)

    # ---- bars 0-4  intro: alone, far out, the disk barely lit
    add(0.0, bar(3), "cold open", mode=0, r=(52, 44), el=(26, 20), az=(6, 20),
        frac=(0.14, 0.17), ty=-0.45, roll=(0.06, 0.01), expo=0.76, disk=0.40, jet=0.06,
        neb=1.05, star=1.10, vign=0.60, grain=0.040, bloom=0.48, sat=0.92,
        disktemp=0.34, intensity=0.18)
    add(bar(3), bar(5), "approach", mode=0, r=(30, 25), el=(-14, -6), az=(150, 168),
        frac=(0.26, 0.32), ty=0.35, roll=(-0.05, 0.02), expo=0.82, disk=0.55, jet=0.10,
        neb=1.00, star=1.05, vign=0.56, grain=0.036, bloom=0.52, sat=0.95,
        disktemp=0.30, intensity=0.25)
    # ---- bar 5  riser into the drop
    add(bar(5), bar(6), "riser / rush in", mode=0, r=(28, 12.5), el=(-6, 6), az=(168, 196),
        frac=(0.30, 0.58), ty=(0.35, 0), expo=(0.82, 1.02), disk=(0.55, 1.05), jet=(0.10, 0.32),
        neb=0.95, vign=(0.56, 0.46), bloom=(0.52, 0.85), intensity=0.55, build=True)

    # ---- bars 6-10  drop A: five one-bar shots, hard cuts
    add(bar(6), bar(7), "close edge-on", r=(11.5, 10.2), el=(5, 9), az=(52, 68), frac=0.60,
        ty=0.28, roll=-0.06, micro=True)
    add(bar(7), bar(8), "high angle, spiral", r=(18.5, 16.5), el=(36, 30), az=(122, 138),
        frac=0.30, ty=-0.38, roll=0.05)
    add(bar(8), bar(9), "under the rim", r=(9.6, 8.8), el=(-4, -9), az=(206, 224), frac=0.70,
        ty=0.20, roll=0.09, micro=True)
    add(bar(9), bar(10), "wide, from below", r=(23, 20), el=(-24, -17), az=(300, 318),
        frac=0.24, ty=0.0, roll=-0.05)
    add(bar(10), bar(11), "long orbit", r=(14.5, 12.5), el=(12, 20), az=(10, 34), frac=0.44,
        ty=0.22, roll=0.04, micro=True)

    # ---- bars 11-16  section B (hats enter): warp bursts between shots
    add(bar(11), bar(12), "warp burst", mode=1, warp=(0.70, 1.00), neb=1.1, star=1.0,
        bloom=0.7, vign=0.40)
    add(bar(12), bar(13), "orbit low", r=(13.5, 11.8), el=(9, 15), az=(60, 78), frac=0.50,
        ty=-0.28, roll=0.04, micro=True)
    add(bar(13), bar(14), "disk wall", r=(9.4, 8.6), el=(8, 3), az=(95, 114), frac=0.66,
        ty=0.24, roll=-0.07)
    add(bar(14), bar(15), "warp burst", mode=1, warp=(1.00, 0.72), bloom=0.75, vign=0.40)
    add(bar(15), bar(16), "top down", r=(20, 17.5), el=(44, 37), az=(190, 208), frac=0.27,
        ty=0.0, roll=0.06, micro=True)
    add(bar(16), beat(67), "grazing the disk", r=(10.5, 9.4), el=(-8, -12), az=(258, 272),
        frac=0.55, ty=-0.18, roll=-0.04)
    # the track drops out for exactly one beat before the breakdown - cut to near black
    add(beat(67), bar(17), "silent beat", r=9.4, el=-12, az=272, frac=0.55, ty=-0.18,
        expo=0.10, disk=0.5, jet=0.0, neb=0.5, star=0.6, bloom=0.3, vign=0.75,
        intensity=0.0, grain=0.05)

    # ---- bars 17-18  breakdown: pull away, cold and quiet
    add(bar(17), bar(19), "breakdown pull-out", r=(26, 50), el=(8, 28), az=(300, 336),
        frac=(0.34, 0.13), ty=(0.2, -0.55), roll=(0.0, 0.07), expo=(0.72, 0.60),
        disk=(0.70, 0.40), sat=(0.92, 0.70), vign=(0.52, 0.64), jet=0.05, neb=1.05,
        bloom=0.46, grain=0.045, intensity=0.22)
    # ---- bar 19  re-entry build
    add(bar(19), bar(20), "re-entry", r=(50, 12.5), el=(28, 8), az=(336, 358),
        frac=(0.13, 0.58), ty=(-0.55, 0), expo=(0.60, 1.02), disk=(0.40, 1.15), jet=(0.05, 0.32),
        vign=(0.64, 0.46), bloom=(0.46, 0.88), intensity=0.65, build=True)

    # ---- bars 20-29  main run: one cut per bar, micro-cuts on the half bar
    add(bar(20), bar(21), "hero close",   r=(12.0, 10.6), el=(6, 11), az=(30, 48), frac=0.58, ty=0.24, roll=-0.05, micro=True)
    add(bar(21), bar(22), "below, rising", r=(17, 15), el=(-28, -20), az=(100, 118), frac=0.32, ty=-0.28, roll=0.06)
    add(bar(22), bar(23), "warp burst", mode=1, warp=(0.85, 1.05), bloom=0.75, vign=0.40)
    add(bar(23), bar(24), "inside the glow", r=(9.6, 8.8), el=(10, 5), az=(170, 190), frac=0.66, ty=0.18, roll=0.08, micro=True)
    add(bar(24), bar(25), "peak - wide tilt", r=(14.5, 12.8), el=(28, 34), az=(240, 256), frac=0.40, ty=-0.32, roll=-0.06, disk=1.12, micro=True)
    add(bar(25), bar(26), "skim the rim", r=(10.5, 9.4), el=(-6, -1), az=(330, 348), frac=0.54, ty=0.15, roll=0.05)
    add(bar(26), bar(27), "warp burst", mode=1, warp=(1.05, 0.80), bloom=0.78, vign=0.40)
    add(bar(27), bar(28), "top of the disk", r=(22, 19), el=(48, 40), az=(60, 78), frac=0.25, ty=0.0, roll=0.07)
    add(bar(28), bar(29), "max close", r=(8.6, 7.8), el=(6, 12), az=(140, 159), frac=0.68, ty=0.26, roll=-0.09, micro=True)
    add(bar(29), bar(30), "drift out", r=(18, 21.5), el=(15, 8), az=(220, 238), frac=0.30, ty=-0.18, roll=0.04)

    # ---- bars 30-32  the bass is filtered out: inside the singularity
    add(bar(30), bar(32), "inside / filaments", mode=2, tunnel=(0.45, 0.85), expo=(0.88, 0.96),
        sat=0.96, grain=0.05, vign=0.54, bloom=0.60, intensity=0.5)
    add(bar(32), bar(33), "tunnel accelerate", mode=2, tunnel=(0.85, 1.20), expo=(0.96, 1.05),
        bloom=0.66, vign=0.50, intensity=0.75, build=True)

    # ---- bars 33-40  finale: the dive
    add(bar(33), bar(34), "back outside", mode=0, r=(28, 20), el=(14, 10), az=(0, 20),
        frac=(0.22, 0.34), ty=-0.20, disk=1.15, jet=0.40, roll=-0.04)
    add(bar(34), bar(35), "committed", r=(16, 11.5), el=(8, 4), az=(42, 60), frac=(0.42, 0.62),
        ty=0.15, disk=1.20, jet=0.45, roll=0.05, micro=True)
    add(bar(35), bar(36), "warp dive", mode=1, warp=(0.95, 1.20), bloom=0.80, vign=0.38)
    add(bar(36), bar(37), "photon ring", r=(8.8, 8.0), el=(9, 4), az=(120, 142), frac=0.68,
        ty=0.18, disk=1.25, jet=0.5, roll=-0.07, micro=True)
    add(bar(37), bar(38), "last light", r=(7.6, 6.8), el=(-7, -3), az=(162, 184), frac=0.72,
        ty=0.10, disk=1.10, jet=0.55, roll=0.09)
    add(bar(38), bar(39), "crossing", mode=1, warp=(1.25, 1.50), bloom=0.85, vign=0.36,
        text=1, talpha=(0.0, 0.95), tscale=0.85, toff=0.30)
    add(bar(39), bar(40), "beyond the horizon", mode=2, tunnel=(1.15, 1.45), bloom=0.74,
        expo=1.0, text=1, talpha=(0.95, 0.0), tscale=0.85, toff=0.30)
    add(bar(40), END, "collapse / singularity", mode=2, tunnel=(1.45, 2.05), bloom=0.82,
        expo=(1.02, 0.95), grain=0.055, intensity=0.9)

    # ---------------------------------------------------- flash / cut events
    flashes = [(bar(6), 0.95), (bar(11), 0.55), (bar(14), 0.55), (bar(17), 0.30),
               (bar(20), 0.95), (bar(22), 0.50), (bar(24), 0.60), (bar(26), 0.50),
               (bar(30), 0.70), (bar(33), 0.85), (bar(35), 0.60), (bar(38), 0.80),
               (bar(39), 0.90), (bar(40), 0.75)]

    # ------------------------------------------------------------- per frame
    nfr = int(math.floor(dur*FPS))
    rows = []
    t_env = np.arange(len(rms))/efps
    for fi in range(nfr):
        t = fi/FPS
        P = dict(DEFAULT)
        P['time'] = t
        sh = None
        for s in S:
            if s.t0 <= t < s.t1:
                sh = s
                break
        if sh is None:
            sh = S[-1]
        p = (t - sh.t0)/max(sh.t1 - sh.t0, 1e-6)
        inten = sh.kw.get('intensity', 1.0)

        # ---- music features at this instant
        i_env = min(int(t*efps), len(rms)-1)
        E = float(rms[i_env])
        hi = float(env_hi[min(i_env, len(env_hi)-1)])

        # ---- beat punch: exponential decay from every beat, weighted by onset
        bi = np.searchsorted(gbeats, t, side='right') - 1
        punch = 0.0
        for j in (bi, bi-1):
            if 0 <= j < len(gbeats):
                dt = t - gbeats[j]
                if 0 <= dt < 0.45:
                    punch += ps[j]*math.exp(-dt/0.105)
        punch = min(punch, 1.4)*inten

        # ---- camera
        mode = sh.kw.get('mode', 0)
        if mode == 0:
            r = sh.val('r', p, 20.0)
            el = sh.val('el', p, 12.0)
            az = sh.val('az', p, 0.0)
            # micro-cut: step the orbit on every other beat inside the shot
            if sh.kw.get('micro'):
                nb_in = int((t - sh.t0)/(period*2))
                if nb_in > 0:
                    az += 21.0*nb_in
                    r *= (1.0 - 0.055*nb_in)
                    el += 4.0*nb_in
            r *= 1.0/(1.0 + 0.055*punch)              # punch = shove toward the hole
            cx, cy, cz = spherical(r, el, az)
            P.update(cx=cx, cy=cy, cz=cz, tx=0.0, ty=sh.val('ty', p, 0.0), tz=0.0)
            frac = sh.val('frac', p, None)
            P['fov'] = fov_for(r, frac) if frac is not None else sh.val('fov', p, 48.0)
            # the closer the camera, the more of the frame the disk fills: hold
            # the exposure roughly constant so close-ups do not blow out
            dist_comp = min(1.20, max(0.45, (r/12.0)**0.55))
        else:
            P.update(cx=0, cy=0, cz=-20, tx=0, ty=0, tz=0)
            P['fov'] = sh.val('fov', p, 48.0)
            dist_comp = 1.0
        P['mode'] = mode
        P['roll'] = sh.val('roll', p, 0.0) + 0.010*math.sin(t*1.7)*inten

        # ---- shake + punch FX
        amp = (0.0010 + 0.0075*punch)*inten*(0.55 + 0.9*E)
        P['shx'] = amp*(0.72*math.sin(t*63.0) + 0.28*math.sin(t*101.0 + 1.3))
        P['shy'] = amp*(0.72*math.sin(t*71.0 + 0.7) + 0.28*math.sin(t*117.0))
        P['roll'] += 0.020*punch*math.sin(t*37.0)*inten
        P['zoom'] = 1.0 + 0.085*punch
        P['rblur'] = min(0.95, 0.80*punch)
        P['chroma'] = 0.0012 + 0.0060*punch + 0.0035*hi*inten

        # ---- look
        P['expo'] = sh.val('expo', p, 1.0)*(1.0 + 0.10*punch)
        P['disk'] = sh.val('disk', p, 1.0)*(1.0 + 0.50*punch)*dist_comp
        P['jet'] = sh.val('jet', p, 0.25)*(1.0 + 0.90*punch)
        P['neb'] = sh.val('neb', p, 1.0)
        P['star'] = sh.val('star', p, 1.0)
        P['twk'] = 0.14 + 0.26*hi
        P['bloom'] = sh.val('bloom', p, 0.62)*(1.0 + 0.15*punch)
        P['grain'] = sh.val('grain', p, 0.028)
        P['vign'] = sh.val('vign', p, 0.45)*(1.0 - 0.18*punch)
        P['sat'] = sh.val('sat', p, 1.02)
        P['contrast'] = 1.06 + 0.05*punch
        P['warp'] = sh.val('warp', p, 0.0)
        P['tunnel'] = sh.val('tunnel', p, 0.0)
        P['disktemp'] = sh.val('disktemp', p, 0.18)
        P['seed'] = 17
        P['stardens'] = 1.0
        P['rings'] = 1.0

        # ---- build sections: rising tension
        if sh.kw.get('build'):
            P['chroma'] += 0.010*p*p
            P['rblur'] = max(P['rblur'], 0.55*p*p)
            P['zoom'] += 0.05*p*p
            P['bloom'] *= 1.0 + 0.35*p

        # ---- flash events
        fl = 0.0
        for ft, fa in flashes:
            d = t - ft
            if 0 <= d < 0.5:
                fl = max(fl, fa*math.exp(-d/0.085))
        fl += 0.018*punch*inten
        P['flash'] = fl

        # ---- title cards
        if sh.kw.get('text') is not None and 'text' in sh.kw:
            P['text'] = sh.kw['text']
            P['talpha'] = sh.val('talpha', p, 0.0)
            P['tscale'] = sh.val('tscale', p, 1.0)*(1.0 + 0.020*punch)
            P['toff'] = sh.val('toff', p, 0.0)
        elif 1.8 < t < 11.2:
            a = min(1.0, (t-1.8)/1.4) * (1.0 - max(0.0, (t-9.4)/1.6))
            P['text'] = 0
            P['talpha'] = max(0.0, a)*0.97
            P['tscale'] = 1.0 + 0.012*punch
            P['toff'] = -0.13

        # ---- fades
        fade = 0.0
        if t < 0.55:
            fade = 1.0 - t/0.55
        if t > dur - 1.5:
            fade = max(fade, min(1.0, (t - (dur-1.5))/1.45))
        P['fade'] = fade

        rows.append(",".join(f"{float(P[k]):.5f}" for k in ORDER))

    with open(out_csv, 'w') as f:
        f.write("\n".join(rows) + "\n")

    # ------------------------------------------------------------- shot list
    with open(out_md, 'w') as f:
        f.write(f"# Edit plan - singularity.wav (VUTRA)\n\n")
        f.write(f"- duration **{dur:.2f} s**, **{nfr}** frames @ {FPS} fps, 1080x1920\n")
        f.write(f"- tempo **{bpm:.2f} BPM** (constant-tempo grid, max residual vs tracked beats "
                f"{resid_ms:.1f} ms -> {'locked' if grid_ok else 'tracked'})\n")
        f.write(f"- beat {period*1000:.1f} ms, bar {period*4*1000:.0f} ms, "
                f"{len(gbeats)} beats / {len(downs)} bars\n")
        f.write(f"- every cut below is on a downbeat; punch/flare/shake amplitude comes "
                f"from the measured onset strength of each beat\n\n")
        f.write("| # | in | out | bars | shot | scene |\n|---|----|-----|------|------|-------|\n")
        for i, s in enumerate(S):
            b0 = int(round((s.t0 - downs[0])/(period*4)))
            b1 = int(round((s.t1 - downs[0])/(period*4)))
            scene = {0: "black hole", 1: "warp", 2: "singularity"}[s.kw.get('mode', 0)]
            f.write(f"| {i+1} | {s.t0:6.2f} | {s.t1:6.2f} | {b0:.2f}-{b1:.2f} | {s.label} | {scene} |\n")
        f.write("\n## flash / impact frames\n\n")
        for ft, fa in flashes:
            f.write(f"- {ft:6.2f}s  intensity {fa:.2f}\n")
    print(f"{nfr} frames, {len(S)} shots, bpm {bpm:.2f}, bar {period*4:.4f}s, "
          f"grid residual {resid_ms:.1f} ms")
    cuts = [s.t0 for s in S]
    dev = [min(abs(c - gb) for gb in gbeats) for c in cuts[1:]]
    print(f"cut-to-beat deviation: max {max(dev)*1000:.1f} ms (frame quantisation is "
          f"{1000/FPS/2:.1f} ms)")


main(sys.argv[1], sys.argv[2], sys.argv[3])
