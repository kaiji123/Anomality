"""
anomality app demo soundtrack: a subtle music bed + UI sounds placed on the recording's real events.

Runs in earshot's venv:  ~/tools/earshot/.venv/bin/python scripts/sound.py   (from demo/)
Reads assets/rec/track.js (clicks and keys with their times) and the camera / caption times of compositions/screen.html.
Writes assets/audio/master.wav and sound/cues.json (frames, for `earshot timeline --cues`).
"""
import json, os, re, sys

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from earshot.forge import kit  # noqa: E402

SR, FPS = 48000, 30
TOTAL = 81.7
END = 74.5                      # the end card starts here
KIT_JSON = os.path.join(ROOT, "sound", "kit.json")
KIT_DIR = os.path.join(ROOT, "sound", "takes")
MUSIC_LUFS = -25.5             # absolute: the bed sits well under the UI sounds ("it should be subtle")
SFX_DB = -2.0
MASTER_LUFS = -20.0
PEAK_CEIL = 10 ** (-2.0 / 20)
BEAT = 0.6                      # 100 BPM


def put(buf, t, y, gain_db=0.0):
    i = int(round(t * SR))
    if y.ndim == 1:
        y = np.stack([y, y], 1)
    a, b = max(i, 0), min(i + len(y), len(buf))
    if b > a:
        buf[a:b] += y[a - i:b - i] * 10 ** (gain_db / 20)


def lufs(meter, y):
    v = meter.integrated_loudness(y)
    return v if np.isfinite(v) else -70.0


def limiter_gain(x, look=0.005, release=0.06):
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    a = np.max(np.abs(x), axis=1)
    need = np.minimum(1.0, PEAK_CEIL / np.maximum(a, 1e-9))
    r = int((look + release) * SR)
    return uniform_filter1d(minimum_filter1d(need, size=2 * r + 1), size=r + 1).astype(np.float32)


class Kit:
    def __init__(self):
        self.lock = kit.render(KIT_JSON, KIT_DIR)
        self.n, self.cache, self.cues = {}, {}, []

    def cue(self, buf, role, t, what, pitch=None, extra_db=0.0, log=True):
        r = self.lock["roles"][role]
        takes = [x for x in r["takes"] if pitch is None or x.get("pitch") == pitch]
        k = self.n.get(role, 0); self.n[role] = k + 1
        tk = takes[k % len(takes)]
        if tk["file"] not in self.cache:
            y, _ = sf.read(os.path.join(KIT_DIR, tk["file"]), dtype="float32", always_2d=True)
            self.cache[tk["file"]] = y if y.shape[1] == 2 else np.repeat(y, 2, 1)
        put(buf, kit.place(tk, r, t), self.cache[tk["file"]], r["level_db"] + SFX_DB + extra_db)
        if log:
            self.cues.append({"f": round(t * FPS), "kind": role, "id": what})


def main():
    track = json.loads(open(os.path.join(ROOT, "assets", "rec", "track.js")).read().split("=", 1)[1].rstrip().rstrip(";"))
    html = open(os.path.join(ROOT, "compositions", "screen.html")).read()
    moves = [(float(a), float(d)) for a, d in re.findall(r"move\(([\d.]+), ([\d.]+),", html)]
    caps = [float(a) for a in re.findall(r'cap\("#scr-c\d+", ([\d.]+),', html)]
    n = int(TOTAL * SR)
    meter = pyln.Meter(SR)
    K = Kit()
    fx = np.zeros((n, 2), np.float32)

    # ---- the recording's own events
    busy = []
    for e in track["events"]:
        if e.get("click"):
            K.cue(fx, "click", e["t"], "click"); busy.append(e["t"])
        elif e.get("key") == "Enter":
            K.cue(fx, "key.return", e["t"], "Enter"); busy.append(e["t"])
        elif e.get("key"):
            K.cue(fx, "key", e["t"], "key " + e["key"]); busy.append(e["t"])
    for a in caps:
        K.cue(fx, "land.light", a + 0.05, "caption lands", extra_db=-3); busy.append(a)
    for a, d in moves:
        mid = a + d / 2
        if min(abs(mid - b) for b in busy) > 0.5:
            K.cue(fx, "move", mid, "camera move")
    # ---- the end card (compositions/endcard.html, frames at 30 fps from END)
    K.cue(fx, "whoosh.long", END + 6 / FPS, "colour flood", extra_db=-6)
    K.cue(fx, "pop", END + 76 / FPS, "the mark pops in", extra_db=-4)
    K.cue(fx, "bell", END + 98 / FPS, "the pulse finishes drawing", pitch="G5", extra_db=-3)

    # ---- subtle bed: kalimba root each bar, one soft marimba note per beat, no drums
    mus = np.zeros((n, 2), np.float32)
    ARP = [["A4", "C5", "E5", "C5"], ["F4", "A4", "C5", "A4"], ["G4", "C5", "E5", "C5"], ["G4", "B4", "D5", "B4"]]
    ROOT_NOTE = ["A4", "F4", "C5", "G4"]
    bar = 0
    while bar * 4 * BEAT < TOTAL - 1.0:
        t0, ch = bar * 4 * BEAT, bar % 4
        K.cue(mus, "m.root", t0, "music", pitch=ROOT_NOTE[ch], log=False)
        for b in range(4):
            K.cue(mus, "m.arp", t0 + b * BEAT, "music", pitch=ARP[ch][b], extra_db=-6 if b % 2 else -3, log=False)
        bar += 1
    mus, _ = kit.stereo_width(mus, 0.6)
    mus = mus.astype(np.float32)
    tt = np.arange(n) / SR
    mus *= np.interp(tt, [0, 0.8, TOTAL - 2.5, TOTAL], [0, 1, 1, 0]).astype(np.float32)[:, None]
    mus *= 10 ** ((MUSIC_LUFS - lufs(meter, mus)) / 20)

    # levels per stem, not on the whole mix: normalising the mix lifted the bed to the clicks' level, because the
    # bed plays all the time and the UI sounds are sparse. UI sounds peak at -3 dBFS; the bed sits at a fixed level under them.
    # Tried and dropped: turning the whole mix up (the limiter squashed the clicks under the bed) and pushing only the
    # UI sounds harder (15 dB into the limiter barely raised their loudness: clicks are short). The bed sets the film's
    # loudness, so the level is a balance: UI sounds +4 dB into the limiter, the bed ~8 dB under their peaks.
    fx *= 10 ** (4.0 / 20) / max(np.max(np.abs(fx)), 1e-9)
    fx *= limiter_gain(fx)[:, None]
    mix = fx    # no music: Kai asked for the sound effects only (the bed above is built but left out)
    mix *= limiter_gain(mix)[:, None]
    os.makedirs(os.path.join(ROOT, "assets", "audio"), exist_ok=True)
    sf.write(os.path.join(ROOT, "assets", "audio", "master.wav"), mix, SR, subtype="PCM_24")
    json.dump({"fps": FPS, "events": [{"f": c["f"], "kind": c["kind"], "id": c["id"]} for c in K.cues]},
              open(os.path.join(ROOT, "sound", "cues.json"), "w"), indent=1)
    print(f"master {lufs(meter, mix):.1f} LUFS, music bed {MUSIC_LUFS} LUFS before master gain, {len(K.cues)} cues")


if __name__ == "__main__":
    main()
