#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun bounty roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (snapshots of the chart) and /input/task.json,
writes /output/forecast.json: for each of the top 10 places, a probability for
each game named being at that place, plus a top-10 guess.

Method: empirical rank-transition kernel, averaged over the scored window.
The chart's own history says how ranks move: for every pair of reads, each
game's rank change is recorded by its starting rank band (1-10, 11-30, 31-60,
61+) and the time between the reads. That gives, per band and per horizon,
the distribution of a game's rank change and its chance of still being on the
chart. A game is forecast by shifting its band's distribution by its own
recent rank velocity (shrunk, because velocity only weakly persists:
autocorrelation ~0.22 in the archive) and averaging over the window's hours.
The top band is far stickier than the rest (median |change| 2 ranks in 24h
vs 7-12 lower down), which a single Gaussian misses, so the kernel is used
directly instead of a fitted curve.

Mass deliberately left unnamed at each place covers games the kernel cannot
name: brand-new entrants and fast climbers from deep in the chart.

Nothing is sampled: two runs on the same history are byte-identical.
Standard library only. Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tuned constants (env overrides exist for local tuning) ---
MODEL = os.environ.get("MF_MODEL", "kernel")              # kernel | gauss
DRIFT_SHRINK = float(os.environ.get("MF_SHRINK", "0.3"))  # shrinkage of velocity
DRIFT_WINDOW_H = float(os.environ.get("MF_WINDOW_H", "48"))  # lookback for velocity
DRIFT_CAP = float(os.environ.get("MF_DRIFT_CAP", "1.5"))  # |drift| cap, ranks/hour
GAP_CLOSE = float(os.environ.get("MF_GAP_CLOSE", "0.3"))  # fraction of the playing gap closed per 48h
NAMED_SHRINK = float(os.environ.get("MF_NAMED_SHRINK", "1.0"))  # scale named mass (rest unnamed)
P_FLOOR = float(os.environ.get("MF_P_FLOOR", "0.002"))    # smallest row written
P_CAP = float(os.environ.get("MF_P_CAP", "0.60"))         # largest row written
NAMED_TARGET = float(os.environ.get("MF_NAMED", "0.80"))  # named mass per place
MIN_OBS_VELOCITY = 3
# kernel parameters
BW_H = float(os.environ.get("MF_BW", "12"))              # horizon smoothing, hours
SMOOTH_SD = float(os.environ.get("MF_SMOOTH", "2.0"))     # rank smoothing of the kernel
FLOOR_MIX = float(os.environ.get("MF_FLOOR_MIX", "0.02")) # uniform floor mixed in
DELTA_MAX = 120                                           # |rank change| clipped here
H_GROUPS = [(1, 6), (7, 12), (13, 24), (25, 36), (37, 48), (49, 60), (61, 72)]
# gaussian fallback parameters
SIGMA1 = float(os.environ.get("MF_SIGMA1", "3.2"))
SD_MAX = float(os.environ.get("MF_SD_MAX", "30.0"))

BANDS = [(1, 10), (11, 30), (31, 60), (61, 10 ** 9)]


def band_of(rank):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= rank <= hi:
            return i
    return len(BANDS) - 1


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def band_hazard(rank):
    if rank <= 10:
        return 0.0008
    if rank <= 30:
        return 0.0022
    if rank <= 60:
        return 0.0021
    return 0.0192


def estimate_drift(obs, window_h):
    """Shrunk linear-fit slope of rank vs time (ranks/hour) over the lookback.

    obs: list of (t_hours, rank) sorted by time, t_hours relative to the end.
    """
    recent = [(t, r) for (t, r) in obs if t >= -window_h]
    if len(recent) < MIN_OBS_VELOCITY:
        if len(obs) >= 2:
            (t0, r0), (t1, r1) = obs[-2], obs[-1]
            dt = t1 - t0
            if dt > 0:
                return max(-DRIFT_CAP, min(DRIFT_CAP, ((r1 - r0) / dt) * DRIFT_SHRINK))
        return 0.0
    n = len(recent)
    mt = sum(t for t, _ in recent) / n
    mr = sum(r for _, r in recent) / n
    num = sum((t - mt) * (r - mr) for t, r in recent)
    den = sum((t - mt) ** 2 for t, _ in recent)
    if den <= 0:
        return 0.0
    return max(-DRIFT_CAP, min(DRIFT_CAP, (num / den) * DRIFT_SHRINK))


def gauss_kernel(sd, radius):
    xs = list(range(-radius, radius + 1))
    ws = [math.exp(-0.5 * (x / sd) ** 2) for x in xs]
    tot = sum(ws)
    return xs, [w / tot for w in ws]


def build_kernel(reads):
    """reads: list of (t_hours, {uid: rank}).

    Returns {(band, group_idx): (cdf, survival)} where cdf is a list over the
    Delta grid -DELTA_MAX..DELTA_MAX (cumulative), survival the chance the
    game is still on the chart at that horizon.
    """
    n_grid = 2 * DELTA_MAX + 1
    # Aggregate transitions by (band, dt rounded to whole hours).
    # cell[(band, dt_bin)] = (hist list, absent weight, total weight)
    cells = {}
    n = len(reads)
    for i in range(n):
        ti, ri = reads[i]
        for j in range(i + 1, n):
            tj, rj = reads[j]
            dt = tj - ti
            if dt <= 0 or dt > 200:
                continue
            dt_bin = int(round(dt))
            for uid, r0 in ri.items():
                b = band_of(r0)
                key = (b, dt_bin)
                cell = cells.get(key)
                if cell is None:
                    cell = ([0.0] * n_grid, 0.0, 0.0)
                    cells[key] = cell
                hist, absent, total = cell
                cell = (hist, absent, total + 1.0)
                cells[key] = cell
                if uid in rj:
                    d = int(round(rj[uid] - r0))
                    d = max(-DELTA_MAX, min(DELTA_MAX, d))
                    hist[d + DELTA_MAX] += 1.0
                else:
                    cells[key] = (hist, absent + 1.0, total + 1.0)
    # Smooth the Delta histogram once per cell is too costly; smooth per
    # (band, group) after weighting instead.
    xs, ws = gauss_kernel(SMOOTH_SD, 6)
    out = {}
    for b in range(len(BANDS)):
        for gi, (lo, hi) in enumerate(H_GROUPS):
            h_mid = (lo + hi) / 2.0
            hist = [0.0] * n_grid
            absent = 0.0
            total = 0.0
            for (bb, dt_bin), (ch, ab, tt) in cells.items():
                if bb != b:
                    continue
                w = math.exp(-0.5 * ((dt_bin - h_mid) / BW_H) ** 2)
                if w < 0.01:
                    continue
                absent += w * ab
                total += w * tt
                for k in range(n_grid):
                    if ch[k]:
                        hist[k] += w * ch[k]
            if total <= 0:
                continue
            present = total - absent
            if present <= 0:
                continue
            # Smooth over Delta, then mix a uniform floor.
            sm = [0.0] * n_grid
            for k in range(n_grid):
                if hist[k] == 0.0:
                    continue
                for dx, wgt in zip(xs, ws):
                    kk = k + dx
                    if 0 <= kk < n_grid:
                        sm[kk] += hist[k] * wgt
            tot = sum(sm)
            if tot <= 0:
                continue
            sm = [v / tot for v in sm]
            uni = 1.0 / n_grid
            sm = [(1.0 - FLOOR_MIX) * v + FLOOR_MIX * uni for v in sm]
            cdf = []
            acc = 0.0
            for v in sm:
                acc += v
                cdf.append(acc)
            out[(b, gi)] = (cdf, present / total)
    return out


def cdf_at(cdf, x):
    """CDF of the Delta distribution at continuous x."""
    if x < -DELTA_MAX:
        return 0.0
    if x >= DELTA_MAX:
        return 1.0
    pos = x + DELTA_MAX
    lo = int(math.floor(pos))
    frac = pos - lo
    if lo >= len(cdf) - 1:
        return cdf[-1]
    return cdf[lo] * (1.0 - frac) + cdf[lo + 1] * frac


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    n_hours = int(task.get("window", {}).get("slots", 72))
    out_name = task.get("output", "forecast.json")

    snaps = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    if not snaps:
        raise SystemExit("empty history")
    snaps.sort(key=lambda s: s["at"])
    t_end = parse_time(snaps[-1]["at"])

    obs = {}
    reads = []
    for s in snaps:
        t = (parse_time(s["at"]) - t_end).total_seconds() / 3600.0
        ranks = {}
        for row in s.get("up_and_coming") or []:
            uid = row["universe_id"]
            ranks[uid] = float(row["rank"])
            obs.setdefault(uid, []).append((t, float(row["rank"])))
        reads.append((t, ranks))
    last_chart = reads[-1][1]
    n_chart = max(len(last_chart), 1)

    kernel = build_kernel(reads) if MODEL == "kernel" else {}
    use_kernel = bool(kernel)

    # Playing-implied rank at the last read: a game whose player count suggests
    # a much better rank than it holds tends to climb toward it (and vice
    # versa). Gap = actual rank - implied rank.
    playing_now = {r["universe_id"]: (r.get("playing") or 0)
                   for r in snaps[-1].get("up_and_coming") or []}
    by_play = sorted(last_chart.items(), key=lambda kv: -playing_now.get(kv[0], 0))
    implied = {uid: i + 1 for i, (uid, _) in enumerate(by_play)}

    prob = {uid: [0.0] * (places + 1) for uid in last_chart}
    exp_rank = {}
    for uid, r0 in last_chart.items():
        drift = estimate_drift(obs.get(uid, []), DRIFT_WINDOW_H)
        if GAP_CLOSE:
            gap = r0 - implied.get(uid, r0)
            drift = max(-DRIFT_CAP, min(DRIFT_CAP, drift - GAP_CLOSE * gap / 48.0))
        b = band_of(r0)
        acc = [0.0] * (places + 1)
        er = 0.0
        for gi, (lo, hi) in enumerate(H_GROUPS):
            h_mid = (lo + hi) / 2.0
            w_hours = min(hi, n_hours) - lo + 1
            if w_hours <= 0:
                continue
            if use_kernel and (b, gi) in kernel:
                cdf, surv = kernel[(b, gi)]
                shift = drift * h_mid
                for k in range(1, places + 1):
                    p = cdf_at(cdf, k + 0.5 - r0 - shift)
                    if k > 1:
                        p -= cdf_at(cdf, k - 0.5 - r0 - shift)
                    acc[k] += surv * p * w_hours
            else:
                # Gaussian fallback for this band/horizon.
                surv = (1.0 - band_hazard(r0)) ** h_mid
                mean = r0 + drift * h_mid
                sd = min(SD_MAX, SIGMA1 * math.sqrt(h_mid))
                below = norm_cdf((0.5 - mean) / sd)
                for k in range(1, places + 1):
                    p = norm_cdf((k + 0.5 - mean) / sd) - norm_cdf((k - 0.5 - mean) / sd)
                    if k == 1:
                        p += below
                    acc[k] += surv * p * w_hours
            er += min(max(r0 + drift * h_mid, 1.0), float(n_chart)) * w_hours
        for k in range(1, places + 1):
            prob[uid][k] = acc[k] / n_hours
        exp_rank[uid] = er / n_hours

    rows = []
    for k in range(1, places + 1):
        tot = sum(prob[uid][k] for uid in last_chart)
        scale = (NAMED_TARGET / tot) if tot > NAMED_TARGET else 1.0
        for uid in last_chart:
            p = prob[uid][k] * scale * NAMED_SHRINK
            if p >= P_FLOOR:
                rows.append({"universe_id": int(uid), "place": k,
                             "p": round(min(p, P_CAP), 6)})
    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    top10 = sorted(last_chart, key=lambda u: (exp_rank[u], u))[:places]
    if len(top10) < places:
        rest = sorted(last_chart, key=lambda u: -sum(prob[u][1:]))
        for uid in rest:
            if uid not in top10:
                top10.append(uid)
            if len(top10) == places:
                break

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
        f.write(json.dumps({"probabilities": rows,
                            "top10": [int(u) for u in top10]}) + "\n")


if __name__ == "__main__":
    main()
