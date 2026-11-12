# Roblox Up-and-Coming Top-10 Forecaster

Entry for the make.fun bounty
[Predict Roblox's Up-and-Coming top 10](https://make.fun/bounty/roblox-up-and-coming-forecast-01)
($500 total prize pool).

## What it does

`main.py` reads the chart's history (`/input/history.ndjson`, hourly snapshots
of Roblox's Up-and-Coming chart) and writes one forecast
(`/output/forecast.json`): for each of the top 10 places, a probability for
each game named being at that place, plus a best-guess top 10 in order.

Method: an empirical rank-transition kernel, averaged over the 72-hour scored
window, ensembled 85/15 with the persistence baseline and sharpened with two
playing-count signals: the slope of each game's playing-implied rank (the
playing order moves before the lagging computed rank) and the gap between its
actual rank and its playing-implied rank. The top band is far stickier than
the rest, which a single Gaussian misses, so the kernel is used directly. Mass
is deliberately left unnamed at each place to cover brand-new entrants and
fast climbers the kernel cannot name. Nothing is sampled, so two runs on the
same history are byte-identical.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Requires Python 3.13 (standard library only), no network. Runs in under a
second on the practice history.

## Local validation

Validated walk-forward on the public archive (8 cut points, 86 scored reads):
mean log score about -37.2, versus -38.1 for the kernel alone and -43.5 for the
house persistence baseline on the practice window. On the practice window itself
the entry scores about -35.4 (baseline -43.5, best entries about -33.6).

## License

MIT. The practice history is used under the MIT licence of the public archive
(github.com/Kenny-Dhub/roblox-snapshots) as noted in the kit.
