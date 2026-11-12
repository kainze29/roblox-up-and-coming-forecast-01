#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun bounty roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (snapshots of the chart) and /input/task.json,
writes /output/forecast.json: for each of the top 10 places, a probability for
each game named being at that place, plus a top-10 guess.

Method: empirical rank-transition kernel, averaged over the scored window,
ensembled 85/15 with the persistence baseline, with two playing-count
signals. The chart's own history says how ranks move: for every pair of
reads, each game's rank change is recorded by its starting rank band (1-10,
11-30, 31-60, 61+) and the time between the reads. That gives, per band and
per horizon, the distribution of a game's rank change and its chance of
still being on the chart. A game is forecast by shifting its band's
distribution by its own recent rank velocity (shrunk, because velocity only
weakly persists: autocorrelation ~0.22 in the archive) and averaging over
the window's hours. The top band is far stickier than the rest (median
|change| 2 ranks in 24h vs 7-12 lower down), which a single Gaussian misses,
so the kernel is used directly instead of a fitted curve.

Two signals beyond rank velocity, both validated walk-forward on the public
archive (8 cut points, 86 scored reads) against the kernel alone:
- Playing-implied rank velocity: the chart is sorted by user growth, and
  the playing order moves before the (computed, lagging) rank does, so the
  slope of a game's playing-implied rank is added to its rank velocity
  (shrink 0.15). A snapshot's playing gap (actual rank minus implied rank)
  is also closed at 0.3 per 48h, as before.
- Persistence ensemble: 15% of the house persistence baseline is mixed in.
  The kernel spreads mass thin over 72h of churn; the baseline re-concentrates
  some of it on the chart as it stands, which the early, most predictable
  window hours reward. Walk-forward mean -37.23 vs -38.10 for the kernel
  alone (original 4 cuts -37.14 vs -37.50; 4 fresh holdout cuts -37.31 vs
  -38.70).

Tried and rejected on the same walk-forward (all worse or noise): recency
weighting of kernel pairs, decaying drift persistence, per-place named
mass from measured entrant rates, a newcomer-climb (tenure) bonus (the
diagnostic gap is real but rank velocity already captures it), velocity
windows 24/72/96h, shrink 0/0.15/0.5/0.7, gap-close 0/0.15/0.5/0.7, kernel
bandwidth 6/24/48, smoothing 1/3/4, named mass 0.70/0.90/0.95, and a
Gaussian-only model.

Mass deliberately left unnamed at each place covers games the kernel cannot
name: brand-new entrants and fast climbers from deep in the chart (in the
archive, 12-42% of top-10 slots 48-96h out are games that were not on the
chart at the earlier read).

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
NAMED_BY_PLACE = os.environ.get("MF_NAMED_BY_PLACE", "")  # e.g. "0.9,0.88,..." overrides MF_NAMED per place
BASE_MIX = float(os.environ.get("MF_BASE_MIX", "0.15"))  # mix with persistence baseline
REC_TAU_H = float(os.environ.get("MF_REC_TAU_H", "0"))    # recency half-life-ish tau for kernel pairs (0=off)
DRIFT_TAU_H = float(os.environ.get("MF_DRIFT_TAU_H", "0"))  # drift persistence tau, hours (0=linear)
PLAY_SHRINK = float(os.environ.get("MF_PLAY_SHRINK", "0.15"))  # shrink for implied-rank (playing) velocity
TENURE_A = float(os.environ.get("MF_TENURE_A", "0"))      # new-entrant climb bonus, ranks/hour (0=off)
TENURE_TAU_H = float(os.environ.get("MF_TENURE_TAU_H", "48"))  # tenure decay tau, hours
MIN_OBS_VELOCITY = 3
# kernel parameters
BW_H = float(os.environ.get("MF_BW", "12"))              # horizon smoothing, hours
SMOOTH_SD = float(os.environ.get("MF_SMOOTH", "2.0"))     # rank smoothing of the kernel
FLOOR_MIX = float(os.environ.get("MF_FLOOR_MIX", "0.0")) # uniform floor mixed in
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
            # Recency weight: pairs ending long ago count less (chart regime
            # drifts as games age out of the 28-day window).
            w_rec = 1.0
            if REC_TAU_H > 0:
                w_rec = math.exp(tj / REC_TAU_H)  # tj <= 0, hours before end
            dt_bin = int(round(dt))
            for uid, r0 in ri.items():
                b = band_of(r0)
                key = (b, dt_bin)
                cell = cells.get(key)
                if cell is None:
                    cell = ([0.0] * n_grid, 0.0, 0.0)
                    cells[key] = cell
                hist, absent, total = cell
                cell = (hist, absent, total + w_rec)
                cells[key] = cell
                if uid in rj:
                    d = int(round(rj[uid] - r0))
                    d = max(-DELTA_MAX, min(DELTA_MAX, d))
                    hist[d + DELTA_MAX] += w_rec
                else:
                    cells[key] = (hist, absent + w_rec, total + w_rec)
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
    obs_imp = {}
    first_seen = {}
    reads = []
    for s in snaps:
        t = (parse_time(s["at"]) - t_end).total_seconds() / 3600.0
        ranks = {}
        for row in s.get("up_and_coming") or []:
            uid = row["universe_id"]
            ranks[uid] = float(row["rank"])
            obs.setdefault(uid, []).append((t, float(row["rank"])))
            if uid not in first_seen:
                first_seen[uid] = t
        reads.append((t, ranks))
    # Implied rank per read, from the playing order (for the playing-velocity
    # signal below).
    obs_imp = {}
    for s in snaps:
        t = (parse_time(s["at"]) - t_end).total_seconds() / 3600.0
        rows = s.get("up_and_coming") or []
        by_play = sorted(rows, key=lambda r: -(r.get("playing") or 0))
        for i, r in enumerate(by_play):
            obs_imp.setdefault(r["universe_id"], []).append((t, float(i + 1)))
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
    # Persistence-baseline distribution, for the optional ensemble mix.
    last_order = sorted(last_chart, key=lambda u: last_chart[u])
    base_p = {}
    for idx, uid in enumerate(last_order):
        rk = idx + 1
        for k in range(1, places + 1):
            if k == 1:
                p = 0.5 if rk == 1 else (0.3 if rk == 2 else 0.0)
            else:
                p = 0.5 if rk == k else (0.15 if abs(rk - k) == 1 else 0.0)
            if p:
                base_p[(uid, k)] = p
    for uid, r0 in last_chart.items():
        drift = estimate_drift(obs.get(uid, []), DRIFT_WINDOW_H)
        if TENURE_A:
            # Newcomer climb: games first seen on the chart recently keep
            # climbing (mean -3.2 ranks over 24-96h) while veterans fall
            # (mean +7.7); the band kernel averages both, so give fresh
            # games a drift bonus decaying with their time on the chart.
            tenure_h = -first_seen.get(uid, 0.0)
            drift = max(-DRIFT_CAP, min(DRIFT_CAP,
                                        drift - TENURE_A * math.exp(-tenure_h / TENURE_TAU_H)))
        if PLAY_SHRINK:
            # Playing-implied rank velocity: a game whose player count is
            # climbing the playing order tends to climb the chart too, and
            # playing moves before the (computed, lagging) rank does.
            play_drift = estimate_drift(obs_imp.get(uid, []), DRIFT_WINDOW_H)
            # estimate_drift already applies DRIFT_SHRINK; rescale to PLAY_SHRINK.
            play_drift = play_drift / DRIFT_SHRINK * PLAY_SHRINK if DRIFT_SHRINK else 0.0
            drift = max(-DRIFT_CAP, min(DRIFT_CAP, drift + play_drift))
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
            # Velocity does not persist linearly for 72h: with DRIFT_TAU_H,
            # the effective horizon saturates (drift decays exponentially).
            if DRIFT_TAU_H > 0:
                h_eff = DRIFT_TAU_H * (1.0 - math.exp(-h_mid / DRIFT_TAU_H))
            else:
                h_eff = h_mid
            if use_kernel and (b, gi) in kernel:
                cdf, surv = kernel[(b, gi)]
                shift = drift * h_eff
                for k in range(1, places + 1):
                    p = cdf_at(cdf, k + 0.5 - r0 - shift)
                    if k > 1:
                        p -= cdf_at(cdf, k - 0.5 - r0 - shift)
                    acc[k] += surv * p * w_hours
            else:
                # Gaussian fallback for this band/horizon.
                surv = (1.0 - band_hazard(r0)) ** h_mid
                mean = r0 + drift * h_eff
                sd = min(SD_MAX, SIGMA1 * math.sqrt(h_mid))
                below = norm_cdf((0.5 - mean) / sd)
                for k in range(1, places + 1):
                    p = norm_cdf((k + 0.5 - mean) / sd) - norm_cdf((k - 0.5 - mean) / sd)
                    if k == 1:
                        p += below
                    acc[k] += surv * p * w_hours
            er += min(max(r0 + drift * h_eff, 1.0), float(n_chart)) * w_hours
        for k in range(1, places + 1):
            prob[uid][k] = acc[k] / n_hours
        exp_rank[uid] = er / n_hours

    if BASE_MIX:
        for uid in last_chart:
            for k in range(1, places + 1):
                prob[uid][k] = ((1.0 - BASE_MIX) * prob[uid][k]
                                + BASE_MIX * base_p.get((uid, k), 0.0))

    named_targets = None
    if NAMED_BY_PLACE:
        try:
            named_targets = [float(x) for x in NAMED_BY_PLACE.split(",")]
        except ValueError:
            named_targets = None
    rows = []
    for k in range(1, places + 1):
        tot = sum(prob[uid][k] for uid in last_chart)
        target = (named_targets[k - 1] if named_targets and k - 1 < len(named_targets)
                  else NAMED_TARGET)
        scale = (target / tot) if tot > target else 1.0
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
