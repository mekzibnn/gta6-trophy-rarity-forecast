# Refresh-Aware Earners/Players Trophy Forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mekzibnn/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mekzibnn/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast every GTA VI PS5 trophy's Exophase earn rate at 2026-11-26T00:00:00Z
as five quantiles (5/25/50/75/95%).

## Model

`main.py` (Python 3.13, stdlib only, no network, deterministic) never
forecasts a rate from the last rate. It splits each rate into earners and
players-tracked and forecasts both counts separately, because Exophase's
page refreshes on its own schedule: when it refreshes, the tracked-player
count jumps and every rate is diluted by the newcomers, even though nobody
un-earned a trophy. The forecast is earners-at-target / players-at-target.

- **Players:** a flat history (no refresh seen yet) expects one pending
  refresh of the size the kit documents (638 → 3,868 players). A history
  with refreshes compounds the recent jump ratios over the expected
  further jumps.
- **Earners:** the already-tracked players keep playing (a per-trophy
  share trend fitted on the history, clamped to never fall), plus the
  newcomers earn trophies as they play (a newcomer budget that matures
  with the horizon, split half in proportion to each trophy's share and
  half onto the earliest trophies new players reach first). The trophy
  every player holds stays at 100%.
- **Spread:** logit-scale, combining the disagreement between the two
  allocation shapes, a random-walk term widening with the horizon, and
  the player-count uncertainty.

Calibration priors come from the kit's own published numbers (RULES.txt):
launch night flat at 638 players / 943 earned, first refresh 3,868
players / 4,276 earned. The newcomer maturity rate is set so the model
reproduces that published refresh in aggregate.

## Run

```bash
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
