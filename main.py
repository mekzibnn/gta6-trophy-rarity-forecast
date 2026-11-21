"""Counts-first earners/players forecaster for GTA VI trophy rarity.

Forecasts, for every PS5 trophy of GTA VI, its earn rate on Exophase
at a case's target, as five quantiles (5/25/50/75/95%). Python 3.13,
standard library only, no network, deterministic.

The model, in one paragraph: forecast COUNTS, never rates. Every
rate is earners divided by players-tracked, and the two counts move
for different reasons: players-tracked jumps when Exophase refreshes
(diluting every rate at once), while earners only ever grow, as
players progress through the game. So the two counts get separate
models. Earners per trophy follow a log-time trend fitted on the
read history (a trend on counts is dilution-free; a trend fitted on
rates is corrupted every time the page refreshes, because the rate
falls while the earners rise). Players-tracked follows a log-time
fit blended with the observed refresh-jump ratios, so launch-week
growth decelerates instead of compounding forever. On top of the
trend-implied earners, the players who join before the target earn
trophies as they play: a newcomer budget (newcomers x the cohort's
mean trophies each x a maturity factor over the horizon) split
across trophies by accessibility -- proportional to (earners+0.5)
raised to a concentration that is steep at short horizons (in
their first 90 minutes new players reach only the most accessible
trophies) and relaxes toward proportional as the horizon lengthens
(over days they spread across the game). The spread on the logit
scale combines the allocation-shape uncertainty (the same budget
under a steeper and a flatter concentration), the earners-trend
fit residual, the players-fit residual, and a random-walk term.

Calibration priors come from the kit's own published numbers:
launch night sat flat at 638 players tracked and 943 trophies
earned, then the first refresh showed 3,868 players and 4,276
earned. A history with no refresh yet expects one pending refresh
of that published size; a history with refreshes extrapolates them
instead. The newcomer maturity timescale (87 minutes) is set so
the model reproduces that published first refresh in aggregate
over a 90-minute horizon.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# First refresh, as the kit documents it (launch-night page flat at
# 638 players / 943 earned, refreshed to 3,868 players / 4,276
# earned). Used only as a prior while a case's history shows no
# refresh at all; a history with refreshes extrapolates them instead.
DOC_PLAYERS_BEFORE = 638.0
DOC_PLAYERS_AFTER = 3868.0
DOC_EARNED_BEFORE = 943.0
DOC_EARNED_AFTER = 4276.0

# Newcomer maturity timescale (minutes): the budget reaches
# 1 - exp(-h_eff / TAU_MATURITY), with h_eff the average newcomer's
# exposure (half the horizon). 87 minutes reproduces the published
# first refresh in aggregate over a 90-minute horizon: the
# non-saturated trophies gained 103 earners, and the model budgets
# 3,230 newcomers x 0.0789 non-saturated trophies each x 0.404.
TAU_MATURITY = 87.0
# Concentration of the newcomer allocation: gamma(h) = 1 + GAMMA_AMP
# * exp(-h_eff_days / GAMMA_TAU). Steep (~3.9) at a 90-minute
# horizon, relaxing toward proportional (~1.3) over 4.5 days.
GAMMA_AMP = 3.0
GAMMA_TAU_DAYS = 1.0

WALK_SIGMA = 0.25
TREND_SIGMA_FLOOR = 0.15
TREND_SIGMA_FLAT = 0.20
PLAYERS_SIGMA_FLAT = 0.05
PLAYERS_SIGMA_FLOOR = 0.10
SIGMA_FLOOR = 0.12
SIGMA_CAP = 1.50
SATURATED_SIGMA = 0.02
SATURATED_SHARE = 0.995


def clamp_pct(x):
    """A percent into the range the logit can take."""
    return min(99.99, max(0.01, x))


def logit_pct(pct):
    """logit of a percent, clamped first (0% and 100% have no logit)."""
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    """Percent from a logit value, kept inside (0, 100)."""
    if lg >= 0.0:
        p = 1.0 / (1.0 + math.exp(-lg))
    else:
        e = math.exp(lg)
        p = e / (1.0 + e)
    return 100.0 * p


def round4_half_up(x):
    """Round to 4 decimals, halves up, as the kit's reference does."""
    scaled = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(scaled) if float(scaled).is_integer() else scaled


def minutes_since_epoch(text):
    """ISO-8601 timestamp (with a trailing Z) to minutes since epoch."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 60.0


def read_minutes(case, fallback):
    """Minutes of a case's checkpoint; falls back to its first read."""
    try:
        return minutes_since_epoch(case["at"])
    except (KeyError, TypeError, ValueError):
        return fallback


def ols(xs, ys):
    """Least-squares line; returns (intercept, slope, rmse) or None."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    den = sum((x - mean_x) ** 2 for x in xs)
    if den <= 1e-12:
        return None
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = num / den
    intercept = mean_y - slope * mean_x
    resid = [(y - (intercept + slope * x)) for x, y in zip(xs, ys)]
    rmse = math.sqrt(sum(r * r for r in resid) / n)
    return intercept, slope, rmse


def latest_read(case):
    """The case's last history read (or None)."""
    history = case.get("history") or []
    return history[-1] if history else None


def earners_per_read(case):
    """Per-read (minutes, players, [earners per trophy]).

    Earners are rate * players / 100, rounded to integers with
    largest-remainder so each read's total matches its trophies_earned.
    A trophy never seen in a read gets None for that read.
    """
    trophies = case["trophies"]
    n = len(trophies)
    out = []
    for read in case.get("history") or []:
        players = read.get("players_tracked")
        if players is None or players <= 0:
            continue
        try:
            slot = minutes_since_epoch(read["slot"])
        except (KeyError, TypeError, ValueError):
            continue
        row = read.get("rates") or []
        raw = []
        for i in range(n):
            r = row[i] if i < len(row) else None
            raw.append(None if r is None else r * players / 100.0)
        counts = [0 if v is None else int(math.floor(v + 0.5))
                  for v in raw]
        earned_total = read.get("trophies_earned")
        if earned_total is not None:
            target = int(earned_total)
            drift = target - sum(counts)
            if drift != 0 and abs(drift) <= n:
                order = sorted(
                    range(n),
                    key=lambda i: ((raw[i] or 0.0) - counts[i], -i),
                    reverse=(drift > 0))
                step = 1 if drift > 0 else -1
                for i in order[:abs(drift)]:
                    counts[i] = max(0, counts[i] + step)
        out.append((slot, float(players), counts))
    return out


def players_at_target(case, horizon_min, players_now, origin_min):
    """Forecast players-tracked at the target, and its sigma.

    Flat history (no refresh seen yet): one pending refresh of the
    documented first-refresh size. History with jumps: the geometric
    mean of a log-time fit and a jump-ratio extrapolation, so
    launch-week growth decelerates instead of compounding forever.
    """
    points = []
    seen = None
    for read in case.get("history") or []:
        players = read.get("players_tracked")
        if players is None:
            continue
        if seen is None or players != seen:
            try:
                points.append((minutes_since_epoch(read["slot"]),
                               float(players)))
            except (KeyError, TypeError, ValueError):
                continue
            seen = players
    if len(points) < 2:
        ratio = DOC_PLAYERS_AFTER / DOC_PLAYERS_BEFORE
        return players_now * ratio, PLAYERS_SIGMA_FLAT
    target_min = read_minutes(case, origin_min) + horizon_min
    # Log-time fit: log P against log1p(hours since release).
    xs = [math.log1p(max(0.0, (t - origin_min) / 60.0)) for t, _ in points]
    ys = [math.log(p) for _, p in points]
    fit = ols(xs, ys)
    cand = []
    if fit is not None:
        intercept, slope, rmse = fit
        slope = min(max(slope, 0.0), 3.0)
        x_t = math.log1p(max(0.0, (target_min - origin_min) / 60.0))
        cand.append((math.exp(intercept + slope * x_t), rmse))
    # Jump-ratio extrapolation: compound the recent jump ratios over
    # the expected number of further jumps.
    ratios = []
    gaps = []
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        if p0 > 0 and p1 > p0 and t1 > t0:
            ratios.append(math.log(p1 / p0))
            gaps.append(t1 - t0)
    if ratios and gaps:
        recent = ratios[-3:]
        rho = math.exp(sum(recent) / len(recent))
        gaps_sorted = sorted(gaps)
        gap = gaps_sorted[len(gaps_sorted) // 2]
        jumps = min(max(horizon_min / gap if gap > 0 else 0.0, 0.0), 40.0)
        # Cap before blending: compounding the jump ratio over many
        # jumps would otherwise drag the geometric mean to the cap
        # even when the log-time fit says growth is decelerating.
        cand.append((min(players_now * (rho ** jumps),
                         players_now * 30.0), None))
    if not cand:
        return players_now, PLAYERS_SIGMA_FLOOR
    vals = [c[0] for c in cand]
    forecast = math.exp(sum(math.log(max(v, 1.0)) for v in vals) / len(vals))
    forecast = min(max(forecast, players_now), players_now * 30.0)
    rmse = [c[1] for c in cand if c[1] is not None]
    sigma = max(PLAYERS_SIGMA_FLOOR, rmse[0]) if rmse else PLAYERS_SIGMA_FLOOR
    if len(vals) > 1:
        disagree = abs(math.log(vals[0] / vals[1])) / 2.0
        sigma = math.sqrt(sigma ** 2 + disagree ** 2)
    return forecast, min(sigma, 0.60)


def earners_trend(reads, idx, origin_min, target_min, earners_now):
    """A trophy's earners at the target from its count trend, and sigma.

    Least-squares line of log(earners) against log1p(hours since
    release). Earners only grow, so the fit is clamped to never fall
    below today's count and never more than 40x it. A flat history
    gives a flat line (exactly zero movement there); the spread
    floor covers the future being uncertain anyway.
    """
    xs = []
    ys = []
    for slot, _, counts in reads:
        e = counts[idx]
        if e is None or e <= 0:
            continue
        xs.append(math.log1p(max(0.0, (slot - origin_min) / 60.0)))
        ys.append(math.log(float(e)))
    if len(xs) < 2:
        return float(earners_now), TREND_SIGMA_FLAT
    fit = ols(xs, ys)
    if fit is None:
        return float(earners_now), TREND_SIGMA_FLAT
    intercept, slope, rmse = fit
    slope = min(max(slope, 0.0), 3.0)
    x_t = math.log1p(max(0.0, (target_min - origin_min) / 60.0))
    value = math.exp(intercept + slope * x_t)
    value = min(max(value, float(earners_now)), float(earners_now) * 40.0)
    sigma = min(max(rmse, TREND_SIGMA_FLOOR), 0.80)
    return value, sigma


def concentration_gamma(horizon_min):
    """Newcomer-allocation concentration for this horizon.

    gamma = 1 + 3 * exp(-h_eff_days), h_eff = half the horizon (the
    average newcomer's exposure). Steep (~3.9) at a 90-minute
    horizon -- in their first hour new players reach only the most
    accessible trophies -- relaxing toward proportional (~1.3) over
    4.5 days, as they spread across the game.
    """
    h_eff_days = (horizon_min / 2.0) / 1440.0
    return 1.0 + GAMMA_AMP * math.exp(-h_eff_days / GAMMA_TAU_DAYS)


def allocate_newcomers(earners_now, budget, gamma, saturated):
    """Split the newcomer budget across trophies by accessibility."""
    weights = []
    for i, e in enumerate(earners_now):
        weights.append(0.0 if saturated[i] else (float(e) + 0.5) ** gamma)
    total = sum(weights) or 1.0
    return [budget * w / total for w in weights]


def forecast_case(case):
    """One case's forecast: {"trophies": [{"id", "q": [5]}]}."""
    trophies = case["trophies"]
    n = len(trophies)
    horizon = float(case.get("horizon_minutes") or 90)
    horizon = max(horizon, 1.0)
    reads = earners_per_read(case)
    last = latest_read(case)
    if reads:
        _, players_now, counts_now = reads[-1]
    elif last is not None and last.get("players_tracked"):
        players_now = float(last["players_tracked"])
        counts_now = [0] * n
    else:
        players_now = DOC_PLAYERS_BEFORE
        counts_now = [0] * n
    rates_last = (last.get("rates") or []) if last else []

    origin_min = minutes_since_epoch(case["game"]["release_date"] + "T00:00:00Z") \
        if case.get("game", {}).get("release_date") else 0.0
    target_min = read_minutes(case, origin_min) + horizon

    saturated = []
    for i in range(n):
        share = (counts_now[i] / players_now) if players_now > 0 else 0.0
        saturated.append(share >= SATURATED_SHARE)

    players_target, players_sigma = players_at_target(
        case, horizon, players_now, origin_min)
    newcomers = max(0.0, players_target - players_now)

    trend = []
    trend_sigma = []
    for i in range(n):
        if saturated[i]:
            trend.append(float(counts_now[i]))
            trend_sigma.append(TREND_SIGMA_FLAT)
        else:
            v, s = earners_trend(reads, i, origin_min, target_min,
                                 counts_now[i])
            trend.append(v)
            trend_sigma.append(s)

    # Newcomer budget: newcomers x the cohort's mean NON-SATURATED
    # trophies each at the target x a maturity factor over the
    # horizon. Saturated trophies are excluded from the mean AND
    # from the allocation: every player already holds them, so a
    # newcomer earning one adds nothing the rate can show (the
    # rate is capped at 99.99 anyway), and counting them in the
    # mean would inflate the budget for the other trophies.
    mean_trophies = (sum(v for v, s in zip(trend, saturated) if not s)
                     / players_target) if players_target > 0 else 0.0
    h_eff = horizon / 2.0
    maturity = 1.0 - math.exp(-h_eff / TAU_MATURITY)
    budget = newcomers * mean_trophies * maturity

    gamma = concentration_gamma(horizon)
    alloc = allocate_newcomers(counts_now, budget, gamma, saturated)
    alloc_lo = allocate_newcomers(counts_now, budget,
                                  max(1.0, gamma - 1.0), saturated)
    alloc_hi = allocate_newcomers(counts_now, budget,
                                  gamma + 1.0, saturated)

    walk_sigma = WALK_SIGMA * math.sqrt(horizon / 1440.0)
    rows = []
    for i in range(n):
        if saturated[i]:
            median = 99.99
            sigma = SATURATED_SIGMA
        else:
            rate = (trend[i] + alloc[i]) / players_target * 100.0
            median = clamp_pct(rate)
            # Spread: the allocation shape's disagreement (the same
            # budget under a steeper and a flatter concentration),
            # the trend fit's residual, the players forecast's
            # uncertainty, and the random walk.
            r_lo = clamp_pct((trend[i] + alloc_lo[i]) / players_target * 100.0)
            r_hi = clamp_pct((trend[i] + alloc_hi[i]) / players_target * 100.0)
            alloc_sigma = abs(logit_pct(r_hi) - logit_pct(r_lo)) / 3.29
            sigma = math.sqrt(alloc_sigma ** 2 + trend_sigma[i] ** 2
                              + players_sigma ** 2 + walk_sigma ** 2)
            sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        centre = logit_pct(median)
        quantiles = []
        prev = 0.01
        for z in Z_SCORES:
            q = round4_half_up(clamp_pct(inv_logit_pct(centre + z * sigma)))
            q = max(q, prev)
            quantiles.append(q)
            prev = q
        rows.append({"id": trophies[i]["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
