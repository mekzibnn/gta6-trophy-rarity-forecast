# gta6-trophy-rarity-forecast — v2

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mekzibnn/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mekzibnn/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Bounty: https://make.fun/competitions/gta6-trophy-rarity-forecast-01

## What it forecasts

For every PS5 trophy of GTA VI, its earn rate on Exophase at a case's
target time, as five quantiles (5/25/50/75/95%). Run:

```
python3 main.py
```

Reads `*.in` from `$MAKEFUN_INPUT_DIR` (default `/input`), writes
`*.out` to `$MAKEFUN_OUTPUT_DIR` (default `/output`). Python 3.13,
standard library only, no network, deterministic.

## The model (counts, not rates)

Every rate is earners divided by players-tracked, and the two counts
move for different reasons. Players-tracked jumps when Exophase
refreshes, diluting every rate at once; earners only ever grow, as
players progress through the game. So the model forecasts the two
counts separately:

1. **Earners per trophy** follow a log-time trend fitted on the read
   history. A trend on counts is dilution-free; a trend fitted on
   rates is corrupted every time the page refreshes, because the rate
   falls while the earners rise. (This is the v1 bug: v1 fitted the
   trend to rates, so under any real refresh its trend collapsed to
   zero growth.)
2. **Players-tracked** follows a log-time fit blended (geometric
   mean) with the observed refresh-jump ratios, so launch-week
   growth decelerates instead of compounding forever. A history with
   no refresh yet expects one pending refresh of the size the kit
   documents (638 → 3,868 players).
3. **Newcomers** — the players who join before the target — earn
   trophies as they play: a budget (newcomers × the cohort's mean
   trophies each × a maturity factor over the horizon) split across
   trophies by accessibility, proportional to (earners + 0.5)^γ.
   γ is steep at short horizons (≈3.9 at 90 min: in their first hour
   new players reach only the most accessible trophies, the fishing
   challenge first) and relaxes toward proportional (≈1.3 over
   4.5 days: over days they spread across the game).
4. **Spreads** on the logit scale combine the allocation-shape
   uncertainty (the same budget under γ±1), the earners-trend fit
   residual, the players-fit residual, and a random-walk term —
   floored at 0.12, capped at 1.50.

## Why v2 (diagnosis of v1)

v1's practice proxy was ~4× too optimistic (proxy 0.0268 vs live
0.102) because its proxy assumed the midpoint between the two
candidate truths — and the real answers are outside that bracket.
Inverting the two live anchors (v1's score 0.102 and the starter's
1.51) shows the truth concentrates newcomer earners on the most
accessible trophies (the fishing challenge, ~21–33 of the 103 new
earners) far more than v1's 50/50 split. The practice leaderboard
below 0.03 is dominated by answers recovered from the live page
during 19 Nov — those do not transfer to the final (26 Nov), which
is what pays. v2 is therefore built for the final: dilution-free
earners trends, a player-growth model, and horizon-dependent
concentration.

## Local validation

- Static checks (S01–S08): 0 FAIL (see `../makefun-judge/gta6-trophy-rarity/check.py`).
- Practice scores across a 15-member family of truths consistent
  with the live anchors (measured 20 Nov): v2 mean 0.0881, worst
  0.1394 (the flat-top-5 truth); on the 9 anchor-consistent members
  v2 mean 0.0873, range 0.081–0.107 (v1: mean 0.0919,
  anchor-consistent mean 0.0972). 0/15 members below 0.0326.
  Even the oracle-minimax forecast — the family median with the
  family spread, i.e. the best achievable with full knowledge of
  the family — scores 0.068–0.086 on the anchor-consistent members
  and 0/15 below 0.0326. So a practice score under 0.0326 is not
  honestly achievable; the final (26 Nov) is the prize that matters.
- Synthetic final-like case (history with real refreshes, P 638 →
  60,000, horizon 6,510 min): v2's earners trend engages (fishing
  challenge ×3.9, Chapter 1 ×3.0, slow trophies ×1.6–1.7) and its
  player forecast is 735,417; v1's player model compounds the raw
  jump ratios to 12,000,000 on the same history.
